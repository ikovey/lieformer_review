"""Slow edge-wise full-admissible tensor product used as the oracle."""

from __future__ import annotations

from collections.abc import Mapping

from torch import Tensor

from .common import FullAdmissibleBase
from .solid_harmonics import center_positions, regular_solid_harmonics
from .tensor_product import component_tensor_product, scatter_sum


class FullAdmissibleEdgeTP(FullAdmissibleBase):
    """Apply every full-admissible outer TP directly on every edge."""

    def forward(
        self,
        features: Mapping[int, Tensor],
        positions: Tensor,
        edge_index: Tensor,
        alpha: Tensor,
        batch: Tensor | None = None,
    ) -> dict[int, Tensor]:
        alpha = self._validate_inputs(features, positions, edge_index, alpha, batch)
        positions = center_positions(positions, batch)
        src, dst = edge_index
        relative = positions.index_select(0, dst) - positions.index_select(0, src)
        edge_geometry = regular_solid_harmonics(relative, self.geometry_orders)
        outputs = self._empty_outputs(positions)

        for path in self.paths:
            source_features = features[path.input_degree].index_select(0, src)
            coupled = component_tensor_product(
                source_features,
                edge_geometry[path.geometry_degree],
                path.input_degree,
                path.geometry_degree,
                path.output_degree,
            )
            message = (coupled @ self.weights.for_path(path.path_id)) * self._path_probe(path)
            message = message * alpha.view(-1, 1, 1)
            outputs[path.output_degree] = outputs[path.output_degree] + scatter_sum(
                message,
                dst,
                positions.shape[0],
            )
        return outputs
