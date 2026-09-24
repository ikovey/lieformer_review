"""Independent SHR-MH model copied from the frozen unified SHR baseline.

Only one state h[N, (lmax+1)^2, C] crosses block boundaries. Degree zero is
the scalar state. This module does not depend on the dual-stream LieFormer or
its adapter, normalization, FFN, or the strict tensor-product implementation.
"""

from __future__ import annotations

import math
import copy
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
from torch_cluster import radius_graph

from ..deps.atomic.cutoff import CosineCutoff
from ..deps.atomic.module_mlp import qMLP
from ..deps.atomic.rbf_expnorm import ExpNormalSmearing
from ..deps.pbc_neighbor import expand_ghost_nodes
from ._shared_primitives import center_positions, regular_solid_harmonics, scatter_sum
from .shr import SparseHarmonicRecoupling
from .attention import ScalarMLPAttention, DotAlphaAttention, neighbor_softmax
from .diagnostics import AttentionDiagnostics


class UnifiedNodeEmbedding(nn.Module):
    """Scalar atom/neighbor embedding with defined empty-neighborhood behavior."""

    def __init__(self, channels, num_basis):
        super().__init__()
        self.atom_embed = nn.Embedding(101, channels)
        self.rbf_projection = qMLP([num_basis, channels], activation=None)
        self.output = qMLP([2 * channels, channels, channels], norm="ln", activation="silu")

    def forward(self, z, edge_index, radial, decay):
        atoms = self.atom_embed(z)
        src, dst = edge_index
        messages = atoms[src] * self.rbf_projection(radial) * decay
        neighbors = scatter_sum(messages, dst, z.shape[0])
        return self.output(torch.cat([atoms, neighbors], dim=-1))


class DegreeRMSNorm(nn.Module):
    """One invariant RMS per node and degree, averaged over m and channels.

    No centering or magnetic-component bias is applied to non-scalar irreps.
    Reductions use at least float32; eps bounds amplification near zero.
    """

    def __init__(self, lmax: int, channels: int, eps: float = 1e-6, shared_high: bool = False):
        super().__init__()
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("norm_eps must be positive")
        self.lmax, self.eps = lmax, eps
        self.shared_high = shared_high
        self.weight = nn.Parameter(torch.ones(lmax + 1, channels))
        self.scalar_bias = nn.Parameter(torch.zeros(channels))

    def forward(self, h: Tensor) -> Tensor:
        parts = []
        shared_inv_rms = None
        if self.shared_high and self.lmax:
            work = h.float() if h.dtype in (torch.float16, torch.bfloat16) else h
            # Equal weight per degree, independent of the number of m components.
            powers = [
                work[:, l * l : (l + 1) ** 2].square().mean(dim=(1, 2), keepdim=True) for l in range(1, self.lmax + 1)
            ]
            shared_inv_rms = torch.rsqrt(torch.stack(powers).mean(dim=0) + self.eps)
        for l in range(self.lmax + 1):
            x = h[:, l * l : (l + 1) * (l + 1)]
            work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
            inv_rms = (
                shared_inv_rms
                if l > 0 and shared_inv_rms is not None
                else torch.rsqrt(work.square().mean(dim=(1, 2), keepdim=True) + self.eps)
            )
            y = (work * inv_rms).to(h.dtype) * self.weight[l]
            if l == 0:
                y = y + self.scalar_bias
            parts.append(y)
        return torch.cat(parts, dim=1)


class UnifiedEqFFN(nn.Module):
    """Return an increment: scalar MLP plus bounded gates on linear irreps."""

    def __init__(self, lmax: int, channels: int, zero_init: bool = True):
        super().__init__()
        self.lmax, self.channels = lmax, channels
        self.hi_proj = nn.Linear(channels, channels, bias=False) if lmax else None
        if self.hi_proj is not None:
            nn.init.xavier_uniform_(self.hi_proj.weight)
        width = (lmax + 1) * channels
        self.scalar_mlp = qMLP([width, 2 * channels, width], norm="ln", activation="silu")
        if zero_init:
            nn.init.zeros_(self.scalar_mlp.layers[-1].weight)
            nn.init.zeros_(self.scalar_mlp.layers[-1].bias)

    def forward(self, h: Tensor) -> Tensor:
        if isinstance(self.hi_proj, nn.ModuleList):
            projected = torch.cat([layer(h[:, l * l : (l + 1) ** 2]) for l, layer in enumerate(self.hi_proj, 1)], dim=1)
        else:
            projected = self.hi_proj(h[:, 1:]) if self.hi_proj is not None else h[:, 1:]
        summaries = [h[:, 0]]
        for l in range(1, self.lmax + 1):
            x = projected[:, l * l - 1 : (l + 1) * (l + 1) - 1]
            # Smooth invariant summaries avoid a singular sqrt at zero.
            summaries.append(torch.sqrt(x.square().mean(dim=1) + 1e-6))
        values = self.scalar_mlp(torch.cat(summaries, dim=-1)).split(self.channels, dim=-1)
        parts = [values[0].unsqueeze(1)]
        for l in range(1, self.lmax + 1):
            x = projected[:, l * l - 1 : (l + 1) * (l + 1) - 1]
            parts.append(x * torch.tanh(values[l]).unsqueeze(1))
        return torch.cat(parts, dim=1)


class SHRMultiHeadBlock(nn.Module):
    def __init__(
        self,
        lmax,
        channels,
        edge_channels,
        *,
        geometry_orders,
        B,
        K,
        norm_eps=1e-6,
        residual_scale=1.0,
        zero_init_ffn=True,
        aggregation_backend="scatter",
    ):
        super().__init__()
        self.lmax = lmax
        self.attention_monitor = None
        self.block_index = None
        self.residual_scale = float(residual_scale)
        self.norm1 = DegreeRMSNorm(lmax, channels, norm_eps)
        self.norm2 = DegreeRMSNorm(lmax, channels, norm_eps)
        self.attention = ScalarMLPAttention(channels, edge_channels, heads=1)
        self.shr = SparseHarmonicRecoupling(
            feature_lmax=lmax,
            in_channels=channels,
            geometry_orders=geometry_orders,
            B=B,
            K=K,
            aggregation_backend=aggregation_backend,
        )
        self.ffn = UnifiedEqFFN(lmax, channels, zero_init=zero_init_ffn)

    def forward(self, h, edge_features, positions, batch, edge_index, ghost2real, geometry):
        normed = self.norm1(h)
        expanded = torch.cat([normed, normed.index_select(0, ghost2real)], dim=0)
        src, dst = edge_index
        logits = self.attention(expanded, edge_features, positions[dst] - positions[src], edge_index)
        alpha = neighbor_softmax(logits, dst, expanded.shape[0])
        if self.attention_monitor is not None:
            self.attention_monitor.capture(self.block_index, logits, alpha, dst, h.shape[0])
        features = {l: expanded[:, l * l : (l + 1) * (l + 1)] for l in range(self.lmax + 1)}
        decoded = self.shr(
            features,
            positions,
            edge_index,
            alpha,
            batch=batch,
            centered_positions=positions,
            node_geometry=geometry,
            num_target_nodes=h.shape[0],
        )
        delta = torch.cat([decoded[l] for l in range(self.lmax + 1)], dim=1)[: h.shape[0]]
        h = h + self.residual_scale * delta
        return h + self.residual_scale * self.ffn(self.norm2(h))


@dataclass
class UnifiedGraph:
    z: Tensor
    positions: Tensor
    batch: Tensor
    edge_index: Tensor
    ghost2real: Tensor
    num_real_nodes: int


class SHRMultiHeadLieFormer(nn.Module):
    """Independent model with interchangeable invariant scoring and grouped SHR."""

    def __init__(
        self,
        dn=256,
        de=256,
        lmax=2,
        n_blocks=8,
        radius=12.0,
        num_basis=128,
        *,
        geometry_orders=(0, 1, 2),
        shr_B=None,
        shr_K=None,
        coordinate_scale=None,
        max_num_neighbors=32,
        pbc_neighbor_backend="torch",
        force_head="direct",
        norm_eps=1e-6,
        residual_scale=1.0,
        zero_init_ffn=True,
        zero_init_output_heads=True,
        activation_checkpoint=False,
        shr_aggregation_backend="scatter",
        high_order_embedding=False,
        embedding_degree_scale=20.0,
        ffn_type="gated",
        s2_grid_resolution=18,
        final_norm=False,
        s2_blocks=None,
        block_norm="degree",
        s2_residual_scope="high",
        s2_residual_gain=0.1,
        s2_residual_cap=0.1,
        s2_residual_eps=1e-12,
        scalar_residual_blocks=None,
        attention_type="scalar_mlp",
        attention_heads=1,
        attention_lmax=None,
        attention_projection=64,
        attention_scalar_head=32,
    ):
        super().__init__()
        if attention_type not in ("scalar_mlp", "dot_alpha"):
            raise ValueError("attention_type must be scalar_mlp or dot_alpha")
        if type(attention_heads) is not int or attention_heads < 1 or dn % attention_heads:
            raise ValueError("attention_heads must be positive and divide dn")
        attention_lmax = lmax if attention_lmax is None else attention_lmax
        if type(attention_lmax) is not int or not 0 <= attention_lmax <= lmax:
            raise ValueError("attention_lmax must be in [0,lmax]")
        if min(attention_projection, attention_scalar_head) < 1:
            raise ValueError("attention widths must be positive")
        self.attention_type, self.attention_heads = attention_type, attention_heads
        self.attention_lmax = attention_lmax
        if ffn_type not in ("gated", "gated_degreewise", "s2", "s2_residual", "scalar_residual"):
            raise ValueError("ffn_type must be gated, gated_degreewise, s2, s2_residual or scalar_residual")
        if min(dn, de, n_blocks, num_basis, max_num_neighbors) <= 0 or lmax < 0:
            raise ValueError("widths, blocks, bases and neighbor limit must be positive; lmax >= 0")
        if block_norm not in ("degree", "shared_high"):
            raise ValueError("block_norm must be degree or shared_high")
        if s2_blocks is not None:
            if ffn_type not in ("s2", "s2_residual"):
                raise ValueError("s2_blocks requires ffn_type=s2 or s2_residual")
            if not isinstance(s2_blocks, (list, tuple)) or not s2_blocks:
                raise ValueError("s2_blocks must be a nonempty list of zero-based block indices")
            if any(type(i) is not int or not 0 <= i < n_blocks for i in s2_blocks):
                raise ValueError("s2_blocks indices must be integers in [0, n_blocks)")
            if len(set(s2_blocks)) != len(s2_blocks):
                raise ValueError("s2_blocks must not contain duplicates")
        if scalar_residual_blocks is not None:
            if ffn_type != "scalar_residual":
                raise ValueError("scalar_residual_blocks requires ffn_type=scalar_residual")
            if not isinstance(scalar_residual_blocks, (list, tuple)) or not scalar_residual_blocks:
                raise ValueError("scalar_residual_blocks must be a nonempty list of zero-based block indices")
            if any(type(i) is not int or not 0 <= i < n_blocks for i in scalar_residual_blocks):
                raise ValueError("scalar_residual_blocks indices must be integers in [0, n_blocks)")
            if len(set(scalar_residual_blocks)) != len(scalar_residual_blocks):
                raise ValueError("scalar_residual_blocks must not contain duplicates")
        if s2_residual_scope not in ("high", "all"):
            raise ValueError("s2_residual_scope must be high or all")
        if not math.isfinite(s2_residual_gain) or s2_residual_gain <= 0:
            raise ValueError("s2_residual_gain must be positive and finite")
        if s2_residual_cap is not None and (not math.isfinite(s2_residual_cap) or s2_residual_cap <= 0):
            raise ValueError("s2_residual_cap must be null or positive and finite")
        if not math.isfinite(s2_residual_eps) or s2_residual_eps <= 0:
            raise ValueError("s2_residual_eps must be positive and finite")
        self.s2_blocks = tuple(
            range(n_blocks) if s2_blocks is None and ffn_type in ("s2", "s2_residual") else (s2_blocks or ())
        )
        self.scalar_residual_blocks = tuple(
            range(n_blocks)
            if scalar_residual_blocks is None and ffn_type == "scalar_residual"
            else (scalar_residual_blocks or ())
        )
        self.block_norm = block_norm
        self.coordinate_scale = float(radius if coordinate_scale is None else coordinate_scale)
        if not math.isfinite(self.coordinate_scale) or self.coordinate_scale <= 0:
            raise ValueError("coordinate_scale must be finite and positive")
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("radius must be finite and positive")
        if not math.isfinite(residual_scale) or residual_scale <= 0:
            raise ValueError("residual_scale must be finite and positive")
        if force_head not in ("direct", "grad") or (force_head == "direct" and lmax < 1):
            raise ValueError("force_head must be direct (requires lmax >= 1) or grad")
        self.dn, self.lmax, self.radius = dn, lmax, float(radius)
        self.geometry_orders = tuple(sorted(set(geometry_orders)))
        self.max_num_neighbors = max_num_neighbors
        self.pbc_neighbor_backend = pbc_neighbor_backend
        self.activation_checkpoint = activation_checkpoint
        self.produces_force = force_head == "direct"
        self.rbf = ExpNormalSmearing(num_basis, cutoff=radius, trainable=False)
        self.cutoff = CosineCutoff(radius)
        self.node_embedding = UnifiedNodeEmbedding(dn, num_basis)
        self.edge_projection = qMLP([num_basis, de], activation=None)
        self.blocks = nn.ModuleList(
            [
                SHRMultiHeadBlock(
                    lmax,
                    dn,
                    de,
                    geometry_orders=self.geometry_orders,
                    B=shr_B,
                    K=shr_K,
                    norm_eps=norm_eps,
                    residual_scale=residual_scale,
                    zero_init_ffn=zero_init_ffn,
                    aggregation_backend=shr_aggregation_backend,
                )
                for _ in range(n_blocks)
            ]
        )
        self.attention_monitor = AttentionDiagnostics()
        for index, block in enumerate(self.blocks):
            block.attention_monitor = self.attention_monitor
            block.block_index = index
        self.energy_head = qMLP([dn, 256, 1], activation="silu")
        self.direct_force_head = nn.Linear(dn, 1, bias=False) if self.produces_force else None
        if zero_init_output_heads:
            nn.init.zeros_(self.energy_head.layers[-1].weight)
            nn.init.zeros_(self.energy_head.layers[-1].bias)
            if self.direct_force_head is not None:
                nn.init.zeros_(self.direct_force_head.weight)

        # Construct the exact baseline first. Optional modules cannot shift the
        # RNG draws of shared attention/SHR/head parameters or the caller's RNG.
        self.high_order_embedding = None
        self.final_norm = DegreeRMSNorm(lmax, dn, norm_eps) if final_norm else nn.Identity()
        self.ffn_type = ffn_type
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed((torch.initial_seed() + 104729) % (2**63))
            if high_order_embedding:
                from .architecture import HighOrderEmbedding

                self.high_order_embedding = HighOrderEmbedding(lmax, dn, num_basis, embedding_degree_scale)
        for i, block in enumerate(self.blocks):
            block.norm1.shared_high = block.norm2.shared_high = block_norm == "shared_high"
            if ffn_type == "gated_degreewise" and lmax:
                # Identical initial function to the shared projection; each
                # degree is then free to learn its own channel map.
                block.ffn.hi_proj = nn.ModuleList([copy.deepcopy(block.ffn.hi_proj) for _ in range(lmax)])
            elif ffn_type == "s2" and i in self.s2_blocks:
                from .architecture import SeparableS2FFN

                with torch.random.fork_rng(devices=[]):
                    torch.random.default_generator.manual_seed((torch.initial_seed() + 130363 + i) % (2**63))
                    block.ffn = SeparableS2FFN(lmax, dn, s2_grid_resolution, zero_init_ffn)
            elif ffn_type == "s2_residual" and i in self.s2_blocks:
                from .architecture import ResidualS2FFN

                with torch.random.fork_rng(devices=[]):
                    torch.random.default_generator.manual_seed((torch.initial_seed() + 130363 + i) % (2**63))
                    block.ffn = ResidualS2FFN(
                        block.ffn,
                        lmax,
                        dn,
                        s2_grid_resolution,
                        include_scalar=s2_residual_scope == "all",
                        gain=s2_residual_gain,
                        relative_cap=s2_residual_cap,
                        eps=s2_residual_eps,
                    )
            elif ffn_type == "scalar_residual" and i in self.scalar_residual_blocks:
                from .architecture import ScalarResidualFFN

                with torch.random.fork_rng(devices=[]):
                    torch.random.default_generator.manual_seed((torch.initial_seed() + 130363 + i) % (2**63))
                    block.ffn = ScalarResidualFFN(
                        block.ffn,
                        lmax,
                        dn,
                        gain=s2_residual_gain,
                        relative_cap=s2_residual_cap,
                        eps=s2_residual_eps,
                    )

        # Build the identical baseline first, then replace scoring modules with
        # isolated RNG streams: neither common tensors nor caller RNG shift.
        if attention_type != "scalar_mlp" or attention_heads != 1:
            for i, block in enumerate(self.blocks):
                with torch.random.fork_rng(devices=[]):
                    torch.random.default_generator.manual_seed((torch.initial_seed() + 196613 + i) % (2**63))
                    block.attention = (
                        ScalarMLPAttention(dn, de, attention_heads)
                        if attention_type == "scalar_mlp"
                        else DotAlphaAttention(
                            dn, de, lmax, attention_heads, attention_projection, attention_scalar_head, attention_lmax
                        )
                    )

    @classmethod
    def from_config(cls, cfg):
        if str(cfg.get("compile_mode", "none")).lower() not in ("none", "off", "false"):
            raise ValueError("Unified SHR currently supports compile_mode=none; use activation_checkpoint for memory")
        shr = cfg.get("shr", {}) or {}
        return cls(
            attention_type=cfg.get("attention_type", "scalar_mlp"),
            attention_heads=cfg.get("attention_heads", 1),
            attention_lmax=cfg.get("attention_lmax", None),
            attention_projection=cfg.get("attention_projection", 64),
            attention_scalar_head=cfg.get("attention_scalar_head", 32),
            dn=cfg.get("dn", 256),
            de=cfg.get("de", 256),
            lmax=cfg.get("lmax", 2),
            n_blocks=cfg.get("n_blocks", 8),
            radius=cfg.get("radius", 12.0),
            num_basis=cfg.get("n_rbf", 128),
            geometry_orders=shr.get("geometry_orders", (0, 1, 2)),
            shr_B=cfg.get("shr_B", shr.get("B", None)),
            shr_K=cfg.get("shr_K", shr.get("K", None)),
            coordinate_scale=shr.get("coordinate_scale", None),
            max_num_neighbors=cfg.get("max_num_neighbors", 32),
            pbc_neighbor_backend=cfg.get("pbc_neighbor_backend", "torch"),
            force_head=cfg.get("force_head", "direct"),
            norm_eps=cfg.get("norm_eps", 1e-6),
            residual_scale=cfg.get("residual_scale", 1.0),
            zero_init_ffn=cfg.get("zero_init_ffn", True),
            zero_init_output_heads=cfg.get("zero_init_output_heads", True),
            activation_checkpoint=cfg.get("activation_checkpoint", False),
            shr_aggregation_backend=cfg.get("shr_aggregation_backend", "scatter"),
            high_order_embedding=cfg.get("high_order_embedding", False),
            embedding_degree_scale=cfg.get("embedding_degree_scale", 20.0),
            ffn_type=cfg.get("ffn_type", "gated"),
            s2_grid_resolution=cfg.get("s2_grid_resolution", 18),
            final_norm=cfg.get("final_norm", False),
            s2_blocks=cfg.get("s2_blocks", None),
            block_norm=cfg.get("block_norm", "degree"),
            s2_residual_scope=cfg.get("s2_residual_scope", "high"),
            s2_residual_gain=cfg.get("s2_residual_gain", 0.1),
            s2_residual_cap=cfg.get("s2_residual_cap", 0.1),
            s2_residual_eps=cfg.get("s2_residual_eps", 1e-12),
            scalar_residual_blocks=cfg.get("scalar_residual_blocks", None),
        )

    def s2_residual_diagnostics(self, reset=False):
        from .architecture import ResidualS2FFN, ScalarResidualFFN

        return {
            str(index): block.ffn.diagnostics(reset=reset)
            for index, block in enumerate(self.blocks)
            if isinstance(block.ffn, (ResidualS2FFN, ScalarResidualFFN))
        }

    def prepare_graph(self, data):
        pos, z, batch = data["pos"], data["z"].long(), data["batch"]
        nr = pos.shape[0]
        if data.get("cell") is not None and data.get("pbc") is not None:
            expanded = expand_ghost_nodes(
                pos.detach(),
                z,
                data["cell"].detach(),
                data["pbc"],
                batch,
                self.radius,
                backend=self.pbc_neighbor_backend,
                max_num_neighbors=self.max_num_neighbors,
            )
            g2r = expanded.ghost2real_index
            shift = expanded.expanded_pos[nr:] - pos.detach().index_select(0, g2r)
            positions = torch.cat([pos, pos.index_select(0, g2r) + shift], dim=0)
            return UnifiedGraph(
                expanded.expanded_z, positions, expanded.expanded_batch, expanded.expanded_edge_index, g2r, nr
            )
        edges = data.get("edge_index")
        if edges is None:
            edges = radius_graph(pos, r=self.radius, batch=batch, max_num_neighbors=self.max_num_neighbors)
        return UnifiedGraph(z, pos, batch, edges, z.new_empty((0,)), nr)

    def embed_graph(self, graph, radial, dist):
        """Embedding seam; returns real-node irreps with unchanged baseline keys."""
        src, dst = graph.edge_index
        scalar = self.node_embedding(graph.z, graph.edge_index, radial, self.cutoff(dist))[: graph.num_real_nodes]
        if self.high_order_embedding is None:
            high = scalar.new_zeros((scalar.shape[0], (self.lmax + 1) ** 2 - 1, self.dn))
        else:
            high = self.high_order_embedding(
                graph.z,
                graph.edge_index,
                graph.positions[dst] - graph.positions[src],
                radial,
                self.cutoff(dist),
                graph.num_real_nodes,
            )
        return torch.cat([scalar.unsqueeze(1), high], dim=1)

    def forward(self, data):
        graph = self.prepare_graph(data)
        src, dst = graph.edge_index
        dist = (graph.positions[dst] - graph.positions[src]).norm(dim=-1, keepdim=True)
        radial = self.rbf(dist)
        h = self.embed_graph(graph, radial, dist)
        edge_features = self.edge_projection(radial)
        positions = center_positions(graph.positions, graph.batch) / self.coordinate_scale
        # Translation splits require all lower degrees even for sparse outer orders.
        geometry = regular_solid_harmonics(positions, max(self.geometry_orders))
        for block in self.blocks:
            if self.activation_checkpoint and self.training:
                h = checkpoint(
                    block,
                    h,
                    edge_features,
                    positions,
                    graph.batch,
                    graph.edge_index,
                    graph.ghost2real,
                    geometry,
                    use_reentrant=False,
                )
            else:
                h = block(h, edge_features, positions, graph.batch, graph.edge_index, graph.ghost2real, geometry)
        return self.readout_features(h, graph)

    def readout_features(self, h, graph):
        """Readout seam; energy and force modules remain separately replaceable."""
        readout = self.final_norm(h)
        atomwise = self.energy_head(readout[:, 0])
        real_batch = graph.batch[: graph.num_real_nodes]
        ngraphs = int(real_batch.max().item()) + 1 if real_batch.numel() else 0
        result = {"pred": scatter_sum(atomwise, real_batch, ngraphs).squeeze(-1), "pred_atomwise": atomwise, "l1": h}
        if self.direct_force_head is not None:
            result["force"] = self.direct_force_head(readout[:, 1:4]).squeeze(-1)
        return result
