"""Sparse Harmonic Recoupling (SHR) operator.

The implementation is deliberately self-contained: it owns path construction,
Wigner-6j recoupling, source-local tensor products, scalar edge aggregation,
and target-local decoding.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

from .paths import OuterPath, SHRBranch, build_outer_paths, recouple_paths
from .aggregation import weighted_aggregate
from .primitives import center_positions, component_tensor_product, regular_solid_harmonics, scatter_sum


@dataclass(frozen=True)
class SHRPlan:
    input_degrees: tuple[int, ...]
    output_degrees: tuple[int, ...]
    geometry_orders: tuple[int, ...]
    outer_paths: tuple[OuterPath, ...]
    branches: tuple[SHRBranch, ...]
    bandwidth: int | None

    def manifest(self) -> dict[str, object]:
        return {
            "input_degrees": list(self.input_degrees),
            "output_degrees": list(self.output_degrees),
            "geometry_orders": list(self.geometry_orders),
            "recoupled_bandwidth_B": self.bandwidth,
            "outer_paths": [[p.input_degree, p.geometry_degree, p.output_degree] for p in self.outer_paths],
            "branches": [
                {"path_id": b.outer_path_id, "l_in": b.input_degree, "ell": b.geometry_degree,
                 "l_out": b.output_degree, "a": b.source_degree, "u": b.target_degree,
                 "f": b.intermediate_degree, "coefficient": b.coefficient}
                for b in self.branches
            ],
        }


class SparseHarmonicRecoupling(torch.nn.Module):
    """Node-centric SHR with arbitrary input and output feature degrees.

    Parameters use the handoff notation: ``B`` is the recoupled spectral
    bandwidth (retaining ``|f-l_out| <= B``), while ``K`` optionally caps the
    outer admissible path set. Set ``B=None`` for the exact full recoupling.
    """

    def __init__(
        self,
        feature_lmax: int | None = None,
        in_channels: int = 1,
        out_channels: int | None = None,
        geometry_orders: Iterable[int] = (0, 1, 2),
        *,
        input_degrees: Iterable[int] | int | None = None,
        output_degrees: Iterable[int] | int | None = None,
        l_in_max: int | None = None,
        l_out_max: int | None = None,
        B: int | None = None,
        recoupled_bandwidth: int | None = None,
        bandwidth: int | None = None,
        K: int | None = None,
        outer_path_subset: Iterable[tuple[int, int, int]] | None = None,
        outer_paths: Iterable[tuple[int, int, int]] | None = None,
        aggregation_backend: str = "scatter",
    ):
        super().__init__()
        if aggregation_backend not in ("scatter", "sparse"):
            raise ValueError("aggregation_backend must be scatter or sparse")
        self.aggregation_backend = aggregation_backend
        if feature_lmax is not None:
            if l_in_max is None: l_in_max = int(feature_lmax)
            if l_out_max is None: l_out_max = int(feature_lmax)
        if input_degrees is None:
            if l_in_max is None: raise ValueError("provide feature_lmax/l_in_max or input_degrees")
            input_degrees = range(int(l_in_max) + 1)
        if output_degrees is None:
            if l_out_max is None: l_out_max = int(l_in_max if l_in_max is not None else max(input_degrees))
            output_degrees = range(int(l_out_max) + 1)
        if isinstance(input_degrees, int): input_degrees = range(input_degrees + 1)
        if isinstance(output_degrees, int): output_degrees = range(output_degrees + 1)
        ins = tuple(sorted(set(int(x) for x in input_degrees)))
        outs = tuple(sorted(set(int(x) for x in output_degrees)))
        geos = tuple(sorted(set(int(x) for x in geometry_orders)))
        if not ins or not outs or not geos or min(ins + outs + geos) < 0:
            raise ValueError("degrees and geometry_orders must be non-empty and non-negative")
        if outer_path_subset is None: outer_path_subset = outer_paths
        bandwidth = recoupled_bandwidth if recoupled_bandwidth is not None else (B if B is not None else bandwidth)
        if bandwidth is not None and int(bandwidth) < 0:
            raise ValueError("B/recoupled_bandwidth must be non-negative or None")
        if out_channels is None: out_channels = in_channels
        if int(in_channels) <= 0 or int(out_channels) <= 0:
            raise ValueError("in_channels and out_channels must be positive")
        paths = build_outer_paths(ins, outs, geos, K=K, outer_path_subset=outer_path_subset)
        recoupled = recouple_paths(paths, B=None if bandwidth is None else int(bandwidth))
        branches = tuple(b for p in paths for b in recoupled[p.path_id])
        self.input_degrees, self.output_degrees, self.geometry_orders = ins, outs, geos
        self.in_channels, self.out_channels = int(in_channels), int(out_channels)
        self.B, self.K = None if bandwidth is None else int(bandwidth), K
        self.paths, self.recoupled_paths = paths, recoupled
        self.plan = SHRPlan(ins, outs, geos, paths, branches, self.B)
        self.path_weights = torch.nn.ParameterDict({str(p.path_id): torch.nn.Parameter(torch.empty(self.in_channels, self.out_channels)) for p in paths})
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = self.in_channels ** -0.5
        for weight in self.path_weights.values(): torch.nn.init.uniform_(weight, -bound, bound)

    def weight_for(self, path: OuterPath | int) -> Tensor:
        return self.path_weights[str(path.path_id if isinstance(path, OuterPath) else path)]

    def symbolic_plan(self) -> SHRPlan: return self.plan
    def moment_table(self) -> tuple[SHRBranch, ...]: return self.plan.branches

    def _validate(self, features: Mapping[int, Tensor], positions: Tensor, edge_index: Tensor, alpha: Tensor, batch: Tensor | None) -> Tensor:
        if positions.ndim != 2 or positions.shape[-1] != 3: raise ValueError("positions must have shape (N,3)")
        n = positions.shape[0]
        for l in self.input_degrees:
            if l not in features: raise KeyError(f"features are missing degree {l}")
            expected = (n, 2*l+1, self.in_channels)
            if tuple(features[l].shape) != expected: raise ValueError(f"degree-{l} features must have shape {expected}")
            if features[l].dtype != positions.dtype or features[l].device != positions.device: raise ValueError("features and positions must share dtype/device")
        if edge_index.ndim != 2 or tuple(edge_index.shape[:1]) != (2,) or edge_index.dtype != torch.long: raise ValueError("edge_index must have shape (2,E) and dtype torch.long")
        if batch is not None:
            if batch.ndim != 1 or batch.shape[0] != n or batch.dtype != torch.long or batch.device != positions.device: raise ValueError("batch must be a long tensor of shape (N,)")
            if edge_index.shape[1] and not torch.equal(batch.index_select(0, edge_index[0]), batch.index_select(0, edge_index[1])): raise ValueError("edges must not connect graphs")
        if alpha.ndim == 2 and alpha.shape[-1] == 1: alpha = alpha[:, 0]
        if alpha.ndim != 1 or alpha.shape[0] != edge_index.shape[1] or alpha.dtype != positions.dtype or alpha.device != positions.device: raise ValueError("alpha must have shape (E,) and match positions dtype/device")
        return alpha

    def _geometry(self, positions: Tensor, node_geometry: Mapping[int, Tensor] | None) -> Mapping[int, Tensor]:
        if node_geometry is not None: return node_geometry
        return regular_solid_harmonics(positions, self.geometry_orders)

    def build_local_moments(self, features: Mapping[int, Tensor], positions: Tensor, edge_index: Tensor, alpha: Tensor, batch: Tensor | None = None, *, centered_positions: Tensor | None = None, node_geometry: Mapping[int, Tensor] | None = None, num_target_nodes: int | None = None) -> dict[tuple[int, int, int], Tensor]:
        alpha = self._validate(features, positions, edge_index, alpha, batch)
        pos = centered_positions if centered_positions is not None else center_positions(positions, batch)
        geom = self._geometry(pos, node_geometry)
        nt = pos.shape[0] if num_target_nodes is None else num_target_nodes
        if not 0 <= nt <= pos.shape[0]:
            raise ValueError("num_target_nodes must select a prefix of positions")
        keys = tuple(sorted({(b.input_degree, b.source_degree, b.intermediate_degree) for b in self.plan.branches}))
        if not keys: return {}
        source_parts, widths = [], []
        for l, a, f in keys:
            src = component_tensor_product(features[l], geom[a], l, a, f)
            source_parts.append(src.reshape(pos.shape[0], -1)); widths.append(src.shape[1] * src.shape[2])
        src_i, dst_i = edge_index
        stacked = torch.cat(source_parts, dim=-1)
        if self.aggregation_backend == "sparse":
            aggregated = weighted_aggregate(stacked, alpha, edge_index, nt)
        else:
            aggregated = scatter_sum(stacked.index_select(0, src_i) * alpha.view(-1, 1), dst_i, nt)
        out, offset = {}, 0
        for key, width in zip(keys, widths):
            f = key[2]
            out[key] = aggregated[:, offset:offset+width].reshape(nt, 2*f+1, self.in_channels)
            offset += width
        return out

    def decode_moments(self, moments: Mapping[tuple[int, int, int], Tensor], positions: Tensor, *, batch: Tensor | None = None, centered_positions: Tensor | None = None, node_geometry: Mapping[int, Tensor] | None = None) -> dict[int, Tensor]:
        pos = centered_positions if centered_positions is not None else center_positions(positions, batch)
        geom = self._geometry(pos, node_geometry)
        outputs = {q: pos.new_zeros((pos.shape[0], 2*q+1, self.out_channels)) for q in self.output_degrees}
        grouped: dict[tuple[int, int, int], dict[tuple[int, int, int], list[tuple[int, float]]]] = defaultdict(lambda: defaultdict(list))
        for b in self.plan.branches:
            grouped[(b.output_degree, b.intermediate_degree, b.target_degree)][(b.input_degree, b.source_degree, b.intermediate_degree)].append((b.outer_path_id, b.coefficient))
        n = pos.shape[0]
        for (q, f, u), terms in grouped.items():
            keys = list(terms)
            stacked_m = torch.cat([moments[k] for k in keys], dim=0)
            g = geom[u]
            stacked_g = g.unsqueeze(0).expand(len(keys), -1, -1).reshape(-1, g.shape[-1])
            dest = component_tensor_product(stacked_m, stacked_g, f, u, q)
            weights = []
            for key in keys:
                w = None
                for pid, coeff in terms[key]: w = coeff * self.weight_for(pid) if w is None else w + coeff * self.weight_for(pid)
                weights.append(w)
            projected = torch.bmm(dest.reshape(len(keys), n*(2*q+1), self.in_channels), torch.stack(weights)).reshape(len(keys), n, 2*q+1, self.out_channels)
            outputs[q] = outputs[q] + projected.sum(dim=0)
        return outputs

    def forward(self, features: Mapping[int, Tensor], positions: Tensor, edge_index: Tensor, alpha: Tensor, batch: Tensor | None = None, *, centered_positions: Tensor | None = None, node_geometry: Mapping[int, Tensor] | None = None, num_target_nodes: int | None = None) -> dict[int, Tensor]:
        self._validate(features, positions, edge_index, alpha, batch)
        pos = centered_positions if centered_positions is not None else center_positions(positions, batch)
        # Ghosts remain source nodes and participate in the original centering;
        # only the target prefix is decoded. No change of origin or path weights.
        moments = self.build_local_moments(features, positions, edge_index, alpha, batch, centered_positions=pos, node_geometry=node_geometry, num_target_nodes=num_target_nodes)
        target_pos = pos[:num_target_nodes]
        target_geometry = None if node_geometry is None else {l: g[:num_target_nodes] for l, g in node_geometry.items()}
        return self.decode_moments(moments, target_pos, centered_positions=target_pos, node_geometry=target_geometry)

    def forward_tuple(self, features: tuple[Tensor, ...], positions: Tensor, edge_index: Tensor, alpha: Tensor, batch: Tensor | None = None) -> tuple[Tensor, ...]:
        fmap = {l: features[i] for i, l in enumerate(self.input_degrees)}
        out = self.forward(fmap, positions, edge_index, alpha, batch=batch)
        return tuple(out[l] for l in self.output_degrees)

    def forward_flat(self, a_hi: Tensor, positions: Tensor, edge_index: Tensor, alpha: Tensor, batch: Tensor | None = None) -> Tensor:
        """Flat-layout adapter for contiguous degrees ``0..max(input_degrees)``.

        Input shape is ``(N, (L+1)^2, C)``; output is packed in the same
        degree-major layout for the configured output degrees.
        """
        if a_hi.ndim != 3 or a_hi.shape[0] != positions.shape[0]:
            raise ValueError("a_hi must have shape (N, components, channels)")
        max_in = max(self.input_degrees)
        if a_hi.shape[1] != (max_in + 1) ** 2 or a_hi.shape[2] != self.in_channels:
            raise ValueError(f"a_hi must have shape (N, {(max_in + 1) ** 2}, {self.in_channels})")
        fmap = {l: a_hi[:, l*l:(l+1)*(l+1), :] for l in self.input_degrees}
        out = self.forward(fmap, positions, edge_index, alpha, batch=batch)
        return torch.cat([out[l] for l in self.output_degrees], dim=1)

    def edgewise_reference(self, features: Mapping[int, Tensor], positions: Tensor, edge_index: Tensor, alpha: Tensor, batch: Tensor | None = None) -> dict[int, Tensor]:
        """Direct edge TP oracle for checking the full (``B=None``) identity."""
        alpha = self._validate(features, positions, edge_index, alpha, batch)
        pos = center_positions(positions, batch)
        src, dst = edge_index
        rel = pos.index_select(0, dst) - pos.index_select(0, src)
        geom = regular_solid_harmonics(rel, self.geometry_orders)
        out = {q: pos.new_zeros((pos.shape[0], 2*q+1, self.out_channels)) for q in self.output_degrees}
        # Apply the shared scalar edge weight before scatter.
        for p in self.paths:
            coupled = component_tensor_product(features[p.input_degree].index_select(0, src), geom[p.geometry_degree], p.input_degree, p.geometry_degree, p.output_degree)
            out[p.output_degree] = out[p.output_degree] + scatter_sum((coupled @ self.weight_for(p)) * alpha.view(-1,1,1), dst, pos.shape[0])
        return out


SHR = SparseHarmonicRecoupling
SHRLayer = SparseHarmonicRecoupling
