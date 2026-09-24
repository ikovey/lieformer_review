"""LieFormer block adapter for the strict tensor-product operators."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import torch
from torch import Tensor

from ..deps.atomic.module_mlp import qMLP
from .deps.path_pruning import OuterPathPolicy
from .factorized import StrictFactorizedFullTP
from .paths import normalize_geometry_orders
from .reference import FullAdmissibleEdgeTP


def resolve_geometry_orders(
    geometry_orders: Iterable[int] | None = None,
    max_geometry_order: int | None = None,
) -> tuple[int, ...]:
    """Resolve the two supported geometry-order config forms without ambiguity."""
    explicit = normalize_geometry_orders(geometry_orders) if geometry_orders is not None else None
    expanded = tuple(range(int(max_geometry_order) + 1)) if max_geometry_order is not None else None
    if expanded is not None and int(max_geometry_order) < 0:
        raise ValueError("max_geometry_order must be non-negative")
    if explicit is not None and expanded is not None and explicit != expanded:
        raise ValueError(
            "strict_zitp.geometry_orders and strict_zitp.max_geometry_order must describe "
            f"the same orders, got {explicit} and {expanded}"
        )
    return explicit or expanded or (0, 1, 2)


class StrictLieConv(torch.nn.Module):
    """Adapt strict degree-separated TPs to LieFormer's flat irrep layout.

    The tensor-product output is strictly equal for ``factorized`` and
    ``edgewise_reference`` when their state dicts are equal. Scalar gating is
    deliberately applied only after the complete strict operator.
    """

    def __init__(
        self,
        lmax: int,
        dn: int,
        *,
        operator: str = "factorized",
        geometry_orders: Iterable[int] | None = None,
        max_geometry_order: int | None = None,
        outer_path_policy: str | dict | OuterPathPolicy = "full_admissible",
        coordinate_scale: float = 1.0,
        probe_path_gates: bool = False,
        learned_path_gates: bool = False,
        learned_gate_budget: float | None = None,
        learned_gate_temperature: float = 1.0,
    ):
        super().__init__()
        self.dn = int(dn)
        self.lmax = int(lmax)
        self.geometry_orders = resolve_geometry_orders(geometry_orders, max_geometry_order)
        self.coordinate_scale = float(coordinate_scale)
        if self.coordinate_scale <= 0:
            raise ValueError(f"strict_zitp.coordinate_scale must be positive, got {coordinate_scale}")

        operator = operator.lower()
        operators = {
            "factorized": StrictFactorizedFullTP,
            "edgewise_reference": FullAdmissibleEdgeTP,
        }
        if operator not in operators:
            raise ValueError(f"Unsupported strict operator {operator!r}; expected one of {tuple(operators)}")
        self.operator_name = operator
        self.strict_tp = operators[operator](
            feature_lmax=self.lmax,
            in_channels=self.dn,
            out_channels=self.dn,
            geometry_orders=self.geometry_orders,
            outer_path_policy=outer_path_policy,
            probe_path_gates=probe_path_gates,
            learned_path_gates=learned_path_gates,
            learned_gate_budget=learned_gate_budget,
            learned_gate_temperature=learned_gate_temperature,
        )
        self.scalar_mix = qMLP([2 * self.dn, 2 * self.dn, 2 * self.dn], norm="ln", activation="silu")
        self._last_profile_stats: dict[str, int | float | str] | None = None

    def _unpack_features(self, a_lo: Tensor, a_hi: Tensor) -> dict[int, Tensor]:
        expected_components = (self.lmax + 1) ** 2
        if tuple(a_hi.shape[1:]) != (expected_components, self.dn):
            raise ValueError(
                f"a_hi must have shape (N, {expected_components}, {self.dn}), got {tuple(a_hi.shape)}"
            )
        features = {
            degree: a_hi[:, degree * degree : (degree + 1) * (degree + 1), :]
            for degree in range(self.lmax + 1)
        }
        features[0] = features[0] + a_lo.unsqueeze(1)
        return features

    def _pack_features(self, features: Mapping[int, Tensor]) -> Tensor:
        return torch.cat([features[degree] for degree in range(self.lmax + 1)], dim=1)

    def forward(
        self,
        a_lo: Tensor,
        a_hi: Tensor,
        node_geom: Tensor | None,
        edge_index: Tensor,
        alpha: Tensor,
        *,
        positions: Tensor | None = None,
        batch: Tensor | None = None,
        centered_positions: Tensor | None = None,
        strict_geometry: Mapping[int, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor]:
        del node_geom  # strict geometry is constructed from positions in one fixed convention
        if positions is None:
            raise ValueError("strict LieConv requires positions")
        if positions.shape[0] != a_hi.shape[0]:
            raise ValueError("positions and strict LieConv features must contain the same nodes")

        strict_positions = positions / self.coordinate_scale
        strict_centered_positions = (
            None
            if centered_positions is None
            else centered_positions / self.coordinate_scale
        )
        strict_geometry = strict_geometry
        if self.operator_name == "edgewise_reference":
            outputs = self.strict_tp(
                self._unpack_features(a_lo, a_hi),
                strict_positions,
                edge_index,
                alpha,
                batch=batch,
            )
        else:
            outputs = self.strict_tp(
                self._unpack_features(a_lo, a_hi),
                strict_positions,
                edge_index,
                alpha,
                batch=batch,
                centered_positions=strict_centered_positions,
                node_geometry=strict_geometry,
            )
        delta_hi = self._pack_features(outputs)

        # Preserve the surrounding LieFormer block design while keeping the
        # nonlinear gate outside the exact recoupling identity.
        summary = delta_hi.norm(dim=1)
        delta_lo, gate = torch.split(
            self.scalar_mix(torch.cat([a_lo, summary], dim=-1)),
            self.dn,
            dim=-1,
        )
        delta_hi = delta_hi * torch.sigmoid(gate).unsqueeze(1)

        self._last_profile_stats = {
            "backend": f"strict_{self.operator_name}",
            "outer_path_policy": self.strict_tp.outer_path_policy,
            "outer_paths": len(self.strict_tp.paths),
            "internal_paths": sum(
                len(paths) for paths in getattr(self.strict_tp, "recoupled_paths", {}).values()
            ),
        }
        return delta_lo, delta_hi
