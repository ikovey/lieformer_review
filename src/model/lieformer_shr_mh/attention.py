"""Invariant edge scoring, independent of SHR aggregation and model readout."""

import math

import torch
from torch import nn
from e3nn import o3
from torch_geometric.utils import softmax

from ..deps.atomic.module_mlp import qMLP


def neighbor_softmax(logits, dst, num_nodes):
    """Each head normalizes over incoming *valid* edges (no padded edges)."""
    return softmax(logits, dst, num_nodes=num_nodes)


class ScalarMLPAttention(qMLP):
    """H=1 retains original attention.layers.* checkpoint keys and numerics."""

    def __init__(self, channels, edge_channels, heads=1):
        super().__init__([2 * channels + edge_channels, 2 * channels, channels, heads], norm="ln", activation="silu")

    def forward(self, features, edge_features, edge_vectors, edge_index):
        src, dst = edge_index
        return super().forward(torch.cat([features[src, 0], features[dst, 0], edge_features], dim=-1))


class DotAlphaAttention(nn.Module):
    """E2Former-style directional contractions with existing SHR radial inputs.

    Both endpoints contract with the SAME source->target unit direction.
    L0-only masks higher degrees BEFORE the learned projections. B/C retain
    identical parameter layouts; disabled projections have zero (not absent)
    gradients. No higher-degree norm is supplied via a separate scoring input.
    """

    def __init__(self, channels, edge_channels, lmax=2, heads=1, projection=64, scalar_head=32, score_lmax=None):
        super().__init__()
        self.lmax, self.heads, self.scalar_head = lmax, heads, scalar_head
        self.score_lmax = lmax if score_lmax is None else score_lmax
        if not 0 <= self.score_lmax <= lmax or min(heads, projection, scalar_head) < 1:
            raise ValueError("invalid dot-alpha degrees or widths")
        self.projections = nn.ModuleList([nn.Linear(channels, projection, bias=(l == 0)) for l in range(lmax + 1)])
        width = 2 * projection * (lmax + 1)
        self.radial = qMLP([edge_channels, edge_channels, width], activation="silu")
        self.fc = nn.Linear(width, heads * scalar_head)
        self.norm = nn.LayerNorm(scalar_head)
        self.alpha_dot = nn.Parameter(torch.empty(heads, scalar_head))
        nn.init.uniform_(self.alpha_dot, -1 / math.sqrt(scalar_head), 1 / math.sqrt(scalar_head))

    def forward(self, features, edge_features, edge_vectors, edge_index):
        src, dst = edge_index
        # clamp prevents undefined coordinate derivatives at coincident nodes.
        unit = edge_vectors / edge_vectors.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        contractions = []
        for l, projection in enumerate(self.projections):
            x = features[:, l * l : (l + 1) ** 2]
            x = x * float(l <= self.score_lmax)
            projected = projection(x)
            y = o3.spherical_harmonics(l, unit, normalize=False, normalization="component")
            contractions.extend(
                [(projected[dst] * y[..., None]).sum(dim=1), (projected[src] * y[..., None]).sum(dim=1)]
            )
        invariant = torch.cat(contractions, dim=-1) * self.radial(edge_features)
        score = self.norm(self.fc(invariant).reshape(-1, self.heads, self.scalar_head))
        score = 0.8 * score * torch.sigmoid(score) + 0.2 * score
        return (score * self.alpha_dot).sum(dim=-1)
