"""Validation and shared setup for strict operators."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor

from .deps.path_pruning import (
    LearnedOuterPathGates,
    OuterPathPolicy,
    OuterPathProbeGates,
    internal_path_costs,
    resolve_path_policy,
)
from .paths import OuterPath, build_outer_paths, normalize_geometry_orders
from .weights import OuterPathWeights


class FullAdmissibleBase(torch.nn.Module):
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
        super().__init__()
        self.feature_lmax = int(feature_lmax)
        self.geometry_orders = normalize_geometry_orders(geometry_orders)
        resolved_policy = resolve_path_policy(outer_path_policy)
        self.outer_path_policy = resolved_policy.name
        self.paths: tuple[OuterPath, ...] = build_outer_paths(
            self.feature_lmax,
            self.geometry_orders,
            resolved_policy,
        )
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.weights = OuterPathWeights(self.paths, self.in_channels, self.out_channels)
        self.path_probe_gates = OuterPathProbeGates(self.paths) if probe_path_gates else None
        if learned_path_gates and probe_path_gates:
            raise ValueError("probe_path_gates and learned_path_gates are mutually exclusive")
        self.learned_path_gates = None
        if learned_path_gates:
            self.learned_path_gates = LearnedOuterPathGates(
                self.paths,
                internal_path_costs(self.feature_lmax, self.geometry_orders),
                target_cost=learned_gate_budget,
                temperature=learned_gate_temperature,
            )

    def _path_probe(self, path: OuterPath) -> Tensor:
        if self.path_probe_gates is None:
            if self.learned_path_gates is None:
                return 1.0
            return self.learned_path_gates.for_path(path)
        return self.path_probe_gates.for_path(path)

    def _validate_inputs(
        self,
        features: Mapping[int, Tensor],
        positions: Tensor,
        edge_index: Tensor,
        alpha: Tensor,
        batch: Tensor | None,
    ) -> Tensor:
        if positions.ndim != 2 or positions.shape[-1] != 3:
            raise ValueError(f"positions must have shape (N, 3), got {tuple(positions.shape)}")
        num_nodes = positions.shape[0]
        for degree in range(self.feature_lmax + 1):
            if degree not in features:
                raise KeyError(f"features are missing degree {degree}")
            expected = (num_nodes, 2 * degree + 1, self.in_channels)
            if tuple(features[degree].shape) != expected:
                raise ValueError(f"degree-{degree} features must have shape {expected}, got {features[degree].shape}")
            if features[degree].dtype != positions.dtype or features[degree].device != positions.device:
                raise ValueError("features and positions must share dtype and device")
        if edge_index.ndim != 2 or tuple(edge_index.shape[:1]) != (2,):
            raise ValueError(f"edge_index must have shape (2, E), got {tuple(edge_index.shape)}")
        if edge_index.dtype != torch.long:
            raise TypeError(f"edge_index must use torch.long, got {edge_index.dtype}")
        if batch is not None and batch.device != positions.device:
            raise ValueError("batch and positions must share a device")
        if batch is not None and edge_index.shape[1] > 0:
            src, dst = edge_index
            if not torch.equal(batch.index_select(0, src), batch.index_select(0, dst)):
                raise ValueError("edges must not connect nodes from different graphs in batch")
        if alpha.ndim == 2 and alpha.shape[1] == 1:
            alpha = alpha[:, 0]
        if alpha.ndim != 1 or alpha.shape[0] != edge_index.shape[1]:
            raise ValueError("phase-1 alpha must be a shared scalar with shape (E,) or (E, 1)")
        if alpha.dtype != positions.dtype or alpha.device != positions.device:
            raise ValueError("alpha and positions must share dtype and device")
        return alpha

    def _empty_outputs(self, positions: Tensor) -> dict[int, Tensor]:
        return {
            degree: positions.new_zeros((positions.shape[0], 2 * degree + 1, self.out_channels))
            for degree in range(self.feature_lmax + 1)
        }
