"""Grouped edge weighting BEFORE path/channel mixing in target decoding."""

import torch
from ._shared_aggregation import weighted_aggregate
from ._shared_primitives import scatter_sum


def grouped_aggregate(features, alpha, edge_index, num_targets, channels, backend="sparse"):
    """Packed layout: [N, sum(moment magnetic widths), H, channels/H].

    Sparse mode processes channel groups without replicating node moments or
    materializing E x total moment-width messages. Path matrices mix channels
    only after this operation. ``block_sparse`` packs heads into disjoint node
    ranges for one sparse aggregation, trading expanded edge indices and layout
    buffers for fewer launches. H=1 uses the original backend exactly.
    """
    if alpha.ndim != 2 or alpha.shape[0] != edge_index.shape[1]:
        raise ValueError("alpha must have shape (E,H)")
    heads = alpha.shape[1]
    if heads < 1 or channels % heads or features.shape[1] % channels:
        raise ValueError("heads must divide channels and packed width must contain full channels")
    if backend not in ("sparse", "scatter", "block_sparse"):
        raise ValueError("unknown aggregation backend")
    src, dst = edge_index
    if heads == 1:
        if backend in ("sparse", "block_sparse"):
            return weighted_aggregate(features, alpha[:, 0], edge_index, num_targets)
        return scatter_sum(features[src] * alpha, dst, num_targets)
    moments = features.shape[1] // channels
    grouped = features.reshape(features.shape[0], moments, heads, channels // heads)
    if backend == "block_sparse":
        n = features.shape[0]
        width_per_head = moments * (channels // heads)
        packed = grouped.permute(2, 0, 1, 3).contiguous().reshape(heads * n, width_per_head)
        offset = torch.arange(heads, device=edge_index.device)[:, None]
        block_edges = torch.stack((
            (src[None, :] + offset * n).reshape(-1),
            (dst[None, :] + offset * num_targets).reshape(-1),
        ))
        result = weighted_aggregate(
            packed, alpha.T.contiguous().reshape(-1), block_edges, heads * num_targets
        )
        return result.reshape(heads, num_targets, moments, channels // heads).permute(
            1, 2, 0, 3
        ).contiguous().reshape(num_targets, features.shape[1])
    results = []
    for head in range(heads):
        part = grouped[:, :, head, :].reshape(features.shape[0], -1)
        weight = alpha[:, head]
        result = (
            weighted_aggregate(part, weight, edge_index, num_targets)
            if backend == "sparse"
            else scatter_sum(part[src] * weight[:, None], dst, num_targets)
        )
        results.append(result.reshape(num_targets, moments, channels // heads))
    return torch.stack(results, dim=2).reshape(num_targets, features.shape[1])
