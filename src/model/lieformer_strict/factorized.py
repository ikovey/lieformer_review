"""Strict source-TP -> scalar-scatter -> destination-TP implementation."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor

from .common import FullAdmissibleBase
from .deps.path_pruning import OuterPathPolicy
from .paths import RecoupledPath, recouple_paths
from .solid_harmonics import center_positions, regular_solid_harmonics
from .tensor_product import component_tensor_product, scatter_sum


class StrictFactorizedFullTP(FullAdmissibleBase):
    """A Wigner-6j factorization exactly equal to ``FullAdmissibleEdgeTP``."""

    def __init__(
        self,
        feature_lmax: int,
        in_channels: int,
        out_channels: int,
        geometry_orders: tuple[int, ...] = (0, 1, 2),
        outer_path_policy: str | dict | OuterPathPolicy = "full_admissible",
        probe_path_gates: bool = False,
        learned_path_gates: bool = False,
        learned_gate_budget: float | None = None,
        learned_gate_temperature: float = 1.0,
    ):
        super().__init__(
            feature_lmax,
            in_channels,
            out_channels,
            geometry_orders,
            outer_path_policy,
            probe_path_gates,
            learned_path_gates,
            learned_gate_budget,
            learned_gate_temperature,
        )
        self.recoupled_paths: dict[int, tuple[RecoupledPath, ...]] = recouple_paths(self.paths)
        self._source_keys = self._build_source_keys()
        self._decode_groups = self._build_decode_groups()

    def _build_source_keys(self):
        keys = {
            (
                branch.input_degree,
                branch.source_geometry_degree,
                branch.intermediate_degree,
            )
            for path in self.paths
            for branch in self.recoupled_paths[path.path_id]
        }
        return tuple(sorted(keys))

    def _build_decode_groups(self):
        """Group branches with the same target-side tensor-product shape."""
        grouped: dict[
            tuple[int, int, int],
            dict[tuple[int, int, int], list[tuple[int, float]]],
        ] = {}
        for path in self.paths:
            for branch in self.recoupled_paths[path.path_id]:
                key = (
                    branch.intermediate_degree,
                    branch.destination_geometry_degree,
                    branch.output_degree,
                )
                moment_key = (
                    branch.input_degree,
                    branch.source_geometry_degree,
                    branch.intermediate_degree,
                )
                grouped.setdefault(key, {}).setdefault(moment_key, []).append(
                    (path.path_id, branch.coefficient)
                )
        return tuple(
            (
                intermediate_degree,
                destination_geometry_degree,
                output_degree,
                tuple(moment_terms.items()),
            )
            for (
                intermediate_degree,
                destination_geometry_degree,
                output_degree,
            ), moment_terms in grouped.items()
        )

    def forward(
        self,
        features: Mapping[int, Tensor],
        positions: Tensor,
        edge_index: Tensor,
        alpha: Tensor,
        batch: Tensor | None = None,
        *,
        centered_positions: Tensor | None = None,
        node_geometry: Mapping[int, Tensor] | None = None,
    ) -> dict[int, Tensor]:
        alpha = self._validate_inputs(features, positions, edge_index, alpha, batch)
        if centered_positions is None:
            positions = center_positions(positions, batch)
        else:
            if centered_positions.shape != positions.shape:
                raise ValueError("centered_positions must have the same shape as positions")
            positions = centered_positions
        if node_geometry is None:
            node_geometry = regular_solid_harmonics(positions, max(self.geometry_orders))
        src, dst = edge_index
        outputs = self._empty_outputs(positions)

        source_parts = []
        source_widths = []
        for input_degree, source_geometry_degree, intermediate_degree in self._source_keys:
            source = component_tensor_product(
                features[input_degree],
                node_geometry[source_geometry_degree],
                input_degree,
                source_geometry_degree,
                intermediate_degree,
            )
            source_parts.append(source.reshape(positions.shape[0], -1))
            source_widths.append(source.shape[1] * source.shape[2])

        # All moments share the same scalar edge weights.  Flattening the
        # source TP outputs allows one gather and one scatter for the whole
        # source stage instead of one pair per recoupling branch.
        stacked_source = torch.cat(source_parts, dim=-1)
        edge_source = stacked_source.index_select(0, src) * alpha.view(-1, 1)
        stacked_moments = scatter_sum(edge_source, dst, positions.shape[0])
        moment_cache: dict[tuple[int, int, int], Tensor] = {}
        offset = 0
        for source_key, width in zip(self._source_keys, source_widths):
            component_count = 2 * source_key[2] + 1
            channel_count = self.in_channels
            moment_cache[source_key] = stacked_moments[:, offset : offset + width].reshape(
                positions.shape[0],
                component_count,
                channel_count,
            )
            offset += width

        for (
            intermediate_degree,
            destination_geometry_degree,
            output_degree,
            moment_terms,
        ) in self._decode_groups:
            moments = []
            combined_weights = []
            for source_key, path_terms in moment_terms:
                moments.append(moment_cache[source_key])
                combined_weight = None
                for path_id, coefficient in path_terms:
                    path = self.paths[path_id]
                    term = coefficient * self.weights.for_path(path_id) * self._path_probe(path)
                    combined_weight = term if combined_weight is None else combined_weight + term
                combined_weights.append(combined_weight)

            node_count = positions.shape[0]
            stacked_moments = torch.cat(moments, dim=0)
            geometry = node_geometry[destination_geometry_degree]
            stacked_geometry = geometry.unsqueeze(0).expand(
                len(moments),
                -1,
                -1,
            ).reshape(-1, geometry.shape[-1])
            stacked_destination = component_tensor_product(
                stacked_moments,
                stacked_geometry,
                intermediate_degree,
                destination_geometry_degree,
                output_degree,
            )
            # One batched GEMM replaces one independent projection per
            # moment. This is important at the production channel width,
            # where many small matmuls otherwise launch serially.
            component_count = 2 * output_degree + 1
            projected = torch.bmm(
                stacked_destination.reshape(len(moments), node_count * component_count, self.in_channels),
                torch.stack(combined_weights, dim=0),
            ).reshape(len(moments), node_count, component_count, self.out_channels)
            outputs[output_degree] = outputs[output_degree] + projected.sum(dim=0)
        return outputs
