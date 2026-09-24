from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

try:
    from ase.neighborlist import primitive_neighbor_list
except ImportError:
    primitive_neighbor_list = None

try:
    from nvalchemiops.torch.neighbors import neighbor_list as alchemi_neighbor_list
except ImportError:
    try:
        from nvalchemiops.neighborlist import neighbor_list as alchemi_neighbor_list
    except ImportError:
        alchemi_neighbor_list = None


def _empty_neighbor_list(device: torch.device):
    return (
        torch.empty((2, 0), dtype=torch.long, device=device),
        torch.empty((0, 3), dtype=torch.long, device=device),
    )


@dataclass(frozen=True)
class ExpandedPBCGraph:
    expanded_pos: Tensor
    expanded_z: Tensor
    expanded_batch: Tensor
    expanded_edge_index: Tensor
    num_real_nodes: int
    ghost2real_index: Tensor


def _concat_neighbor_parts(edge_indices: list[Tensor], offset_list: list[Tensor], device: torch.device):
    if not edge_indices:
        return _empty_neighbor_list(device)
    return torch.cat(edge_indices, dim=1), torch.cat(offset_list, dim=0)


def _normalize_batch_inputs(cell: Tensor, pbc: Tensor, batch: Tensor):
    batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
    if batch_size == 0:
        return batch_size, cell, pbc

    if pbc.dim() == 1:
        pbc = pbc.unsqueeze(0).expand(batch_size, -1)
    if cell.dim() == 2:
        cell = cell.unsqueeze(0).expand(batch_size, -1, -1)
    return batch_size, cell, pbc


def _cell_shifts_to_cartesian(cell_offsets: Tensor, cell: Tensor, edge_batch: Tensor, dtype: torch.dtype) -> Tensor:
    edge_cell = cell.index_select(0, edge_batch)
    return torch.bmm(cell_offsets.to(dtype).unsqueeze(1), edge_cell).squeeze(1)


def _limit_neighbors_reference(
    edge_index: Tensor,
    cell_offsets: Tensor,
    positions: Tensor,
    cell: Tensor,
    batch: Tensor,
    max_num_neighbors: int | None,
):
    if max_num_neighbors is None or max_num_neighbors <= 0 or edge_index.numel() == 0:
        return edge_index, cell_offsets

    if cell.dim() == 2:
        cell = cell.unsqueeze(0)

    src, dst = edge_index
    edge_batch = batch.index_select(0, src)
    shift_vec = _cell_shifts_to_cartesian(cell_offsets, cell, edge_batch, positions.dtype)
    edge_vec = positions.index_select(0, dst) - positions.index_select(0, src) + shift_vec
    edge_dist = torch.norm(edge_vec, dim=1)

    keep_mask = torch.zeros(edge_dist.shape[0], dtype=torch.bool, device=edge_dist.device)
    for node_idx in torch.unique(dst, sorted=True):
        node_edges = torch.nonzero(dst == node_idx, as_tuple=False).squeeze(-1)
        if node_edges.numel() <= max_num_neighbors:
            keep_mask[node_edges] = True
            continue
        topk = torch.topk(
            edge_dist.index_select(0, node_edges),
            k=max_num_neighbors,
            largest=False,
            sorted=False,
        ).indices
        keep_mask[node_edges.index_select(0, topk)] = True

    kept = torch.nonzero(keep_mask, as_tuple=False).squeeze(-1)
    return edge_index.index_select(1, kept), cell_offsets.index_select(0, kept)


def _limit_neighbors(edge_index, cell_offsets, positions, cell, batch, max_num_neighbors):
    """Batched nearest-neighbor selection, preserving the legacy cutoff-tie policy.

    Sort by distance then stably by destination, select each segment's prefix,
    and return edges in their original order. Only segments tied across the
    cutoff need the original per-node topk (whose tie order is unspecified).
    """
    if max_num_neighbors is None or max_num_neighbors <= 0 or edge_index.numel() == 0:
        return edge_index, cell_offsets
    if cell.dim() == 2:
        cell = cell.unsqueeze(0)
    src, dst = edge_index
    shifts = _cell_shifts_to_cartesian(cell_offsets, cell, batch[src], positions.dtype)
    distance = (positions[dst] - positions[src] + shifts).norm(dim=1)
    order = torch.argsort(distance, stable=True)
    order = order[torch.argsort(dst[order], stable=True)]
    sorted_dst = dst[order]
    counts = torch.bincount(dst, minlength=positions.shape[0])
    starts = counts.cumsum(0) - counts
    rank = torch.arange(order.numel(), device=order.device) - starts[sorted_dst]
    keep = torch.zeros_like(dst, dtype=torch.bool)
    keep[order] = rank < max_num_neighbors
    overfull = torch.nonzero(counts > max_num_neighbors).flatten()
    left = order[starts[overfull] + max_num_neighbors - 1]
    right = order[starts[overfull] + max_num_neighbors]
    tied = overfull[distance[left] == distance[right]]
    for node in tied:
        edges = torch.nonzero(dst == node).flatten()
        chosen = distance[edges].topk(max_num_neighbors, largest=False, sorted=False).indices
        keep[edges] = False
        keep[edges[chosen]] = True
    kept = torch.nonzero(keep).flatten()
    return edge_index[:, kept], cell_offsets[kept]


def compute_shifts(cell: Tensor, pbc: Tensor, cutoff: float) -> Tensor:
    reciprocal_cell = cell.inverse().t()
    inv_distances = torch.norm(reciprocal_cell, dim=1)
    num_repeats = torch.ceil(cutoff * inv_distances).to(torch.long)
    num_repeats = torch.where(pbc, num_repeats, torch.zeros_like(num_repeats))

    ranges = []
    for repeats in num_repeats.tolist():
        if repeats <= 0:
            ranges.append(torch.zeros(1, dtype=torch.long, device=cell.device))
        else:
            ranges.append(torch.arange(-repeats, repeats + 1, device=cell.device, dtype=torch.long))

    shifts = torch.cartesian_prod(*ranges)
    if shifts.numel() == 0:
        return torch.zeros((0, 3), dtype=torch.long, device=cell.device)

    keep = (shifts != 0).any(dim=1)
    return shifts[keep]


def _canonical_self_shifts(shifts: Tensor) -> Tensor:
    if shifts.numel() == 0:
        return shifts

    first_nonzero = (shifts != 0) & ((shifts != 0).cumsum(dim=1) == 1)
    keep = ((shifts > 0) & first_nonzero).any(dim=1)
    return shifts[keep]


def build_neighbor_list_single_torch(
    positions: Tensor,
    cell: Tensor,
    pbc: Tensor,
    cutoff: float,
):
    num_atoms = positions.shape[0]
    if num_atoms == 0:
        return _empty_neighbor_list(positions.device)

    device = positions.device
    pair_i_parts = []
    pair_j_parts = []
    pair_offset_parts = []

    base_i, base_j = torch.triu_indices(num_atoms, num_atoms, offset=1, device=device)
    if base_i.numel() > 0:
        pair_i_parts.append(base_i)
        pair_j_parts.append(base_j)
        pair_offset_parts.append(torch.zeros((base_i.shape[0], 3), dtype=torch.long, device=device))

    shifts = compute_shifts(cell, pbc, cutoff)
    if shifts.numel() > 0 and base_i.numel() > 0:
        num_pairs = base_i.shape[0]
        pair_i_parts.append(base_i.repeat(shifts.shape[0]))
        pair_j_parts.append(base_j.repeat(shifts.shape[0]))
        pair_offset_parts.append(shifts.repeat_interleave(num_pairs, dim=0))

    self_shifts = _canonical_self_shifts(shifts)
    if self_shifts.numel() > 0:
        self_atoms = torch.arange(num_atoms, device=device)
        pair_i_parts.append(self_atoms.repeat(self_shifts.shape[0]))
        pair_j_parts.append(self_atoms.repeat(self_shifts.shape[0]))
        pair_offset_parts.append(self_shifts.repeat_interleave(num_atoms, dim=0))

    if not pair_i_parts:
        return _empty_neighbor_list(device)

    pair_i = torch.cat(pair_i_parts, dim=0)
    pair_j = torch.cat(pair_j_parts, dim=0)
    pair_offsets = torch.cat(pair_offset_parts, dim=0)

    shift_values = torch.matmul(pair_offsets.to(cell.dtype), cell)
    edge_vec = positions.index_select(0, pair_j) - positions.index_select(0, pair_i) + shift_values
    distances = torch.norm(edge_vec, dim=1)
    keep = distances < cutoff

    pair_i = pair_i[keep]
    pair_j = pair_j[keep]
    pair_offsets = pair_offsets[keep]

    bi_i = torch.cat([pair_i, pair_j], dim=0)
    bi_j = torch.cat([pair_j, pair_i], dim=0)
    bi_offsets = torch.cat([pair_offsets, -pair_offsets], dim=0)

    order = torch.argsort(bi_i)
    edge_index = torch.stack([bi_i[order], bi_j[order]], dim=0)
    cell_offsets = bi_offsets[order]
    return edge_index, cell_offsets


def build_neighbor_list_batch_torch(
    positions: Tensor,
    cell: Tensor,
    pbc: Tensor,
    batch: Tensor,
    cutoff: float,
):
    batch_size, cell, pbc = _normalize_batch_inputs(cell, pbc, batch)
    if batch_size == 0:
        return _empty_neighbor_list(positions.device)

    edge_indices = []
    offset_list = []
    atom_offset = 0
    for graph_idx in range(batch_size):
        mask = batch == graph_idx
        graph_pos = positions[mask].detach()
        graph_cell = cell[graph_idx].detach()
        graph_pbc = pbc[graph_idx].detach()

        edge_index, cell_offsets = build_neighbor_list_single_torch(graph_pos, graph_cell, graph_pbc, cutoff)
        if edge_index.numel() > 0:
            edge_indices.append(edge_index + atom_offset)
            offset_list.append(cell_offsets)
        atom_offset += int(mask.sum().item())

    return _concat_neighbor_parts(edge_indices, offset_list, positions.device)


def build_neighbor_list_single_ase(
    positions: Tensor,
    cell: Tensor,
    pbc: Tensor,
    cutoff: float,
):
    if primitive_neighbor_list is None:
        raise ImportError("ASE neighbor backend requested, but 'ase' is not installed.")

    if positions.shape[0] == 0:
        return _empty_neighbor_list(positions.device)

    edge_src, edge_dst, shifts = primitive_neighbor_list(
        "ijS",
        pbc=np.asarray(pbc.detach().cpu(), dtype=bool),
        cell=np.asarray(cell.detach().cpu(), dtype=np.float64),
        positions=np.asarray(positions.detach().cpu(), dtype=np.float64),
        cutoff=float(cutoff),
        self_interaction=False,
        use_scaled_positions=False,
    )

    if edge_src.size == 0:
        return _empty_neighbor_list(positions.device)

    edge_index = torch.stack(
        [
            torch.as_tensor(edge_src, dtype=torch.long, device=positions.device),
            torch.as_tensor(edge_dst, dtype=torch.long, device=positions.device),
        ],
        dim=0,
    )
    cell_offsets = torch.as_tensor(shifts, dtype=torch.long, device=positions.device)

    order = torch.argsort(edge_index[0] * max(positions.shape[0], 1) + edge_index[1])
    return edge_index.index_select(1, order), cell_offsets.index_select(0, order)


def build_neighbor_list_batch_ase(
    positions: Tensor,
    cell: Tensor,
    pbc: Tensor,
    batch: Tensor,
    cutoff: float,
):
    batch_size, cell, pbc = _normalize_batch_inputs(cell, pbc, batch)
    if batch_size == 0:
        return _empty_neighbor_list(positions.device)

    edge_indices = []
    offset_list = []
    atom_offset = 0
    for graph_idx in range(batch_size):
        mask = batch == graph_idx
        graph_pos = positions[mask].detach()
        graph_cell = cell[graph_idx].detach()
        graph_pbc = pbc[graph_idx].detach()

        edge_index, cell_offsets = build_neighbor_list_single_ase(graph_pos, graph_cell, graph_pbc, cutoff)
        if edge_index.numel() > 0:
            edge_indices.append(edge_index + atom_offset)
            offset_list.append(cell_offsets)
        atom_offset += int(mask.sum().item())

    return _concat_neighbor_parts(edge_indices, offset_list, positions.device)


def build_neighbor_list_batch_alchemi(
    positions: Tensor,
    cell: Tensor,
    pbc: Tensor,
    batch: Tensor,
    cutoff: float,
    method: str = "cell_list",
):
    if alchemi_neighbor_list is None:
        raise ImportError(
            "ALCHEMI neighbor backend requested, but 'nvalchemi-toolkit-ops[torch]' is not installed."
        )

    batch_size, cell, pbc = _normalize_batch_inputs(cell, pbc, batch)
    if batch_size == 0:
        return _empty_neighbor_list(positions.device)

    kwargs = dict(
        cell=(cell[0] if batch_size == 1 else cell).contiguous(),
        pbc=(pbc[0] if batch_size == 1 else pbc).contiguous(),
        method=method if batch_size == 1 else "batch_" + method,
        return_neighbor_list=False,
    )
    if batch_size > 1:
        kwargs["batch_idx"] = batch.to(torch.int32).contiguous()
    matrix, counts, shifts = alchemi_neighbor_list(positions.contiguous(), float(cutoff), **kwargs)
    # The allocation capacity is NOT the model's nearest-neighbor cap. Find all
    # radius candidates first; never silently lose candidates on dense cells.
    capacity = int(counts.max().item()) if counts.numel() else 0
    if capacity > matrix.shape[1]:
        matrix, counts, shifts = alchemi_neighbor_list(
            positions.contiguous(), float(cutoff), max_neighbors=capacity, **kwargs
        )
        if torch.any(counts > matrix.shape[1]):
            raise RuntimeError("ALCHEMI neighbor allocation overflow after retry")
    row, slot = torch.nonzero(
        torch.arange(matrix.shape[1], device=positions.device)[None, :] < counts[:, None],
        as_tuple=True,
    )
    edge_index = torch.stack([row, matrix[row, slot].long()])
    cell_offsets = shifts[row, slot]

    if edge_index.numel() == 0:
        return _empty_neighbor_list(positions.device)

    edge_index = edge_index.to(dtype=torch.long, device=positions.device)
    cell_offsets = cell_offsets.to(dtype=torch.long, device=positions.device)
    order = torch.argsort(edge_index[0] * max(positions.shape[0], 1) + edge_index[1])
    return edge_index.index_select(1, order), cell_offsets.index_select(0, order)


def build_neighbor_list_batch(
    positions: Tensor,
    cell: Tensor,
    pbc: Tensor,
    batch: Tensor,
    cutoff: float,
    backend: str = "torch",
    max_num_neighbors: int | None = None,
):
    backend = backend.lower()
    if backend == "warp":
        from .warp_neighbor import build_neighbor_list_batch_warp

        return build_neighbor_list_batch_warp(
            positions, cell, pbc, batch, cutoff, max_num_neighbors
        )
    if backend == "torch":
        edge_index, cell_offsets = build_neighbor_list_batch_torch(
            positions=positions, cell=cell, pbc=pbc, batch=batch, cutoff=cutoff
        )
    elif backend == "ase":
        edge_index, cell_offsets = build_neighbor_list_batch_ase(
            positions=positions, cell=cell, pbc=pbc, batch=batch, cutoff=cutoff
        )
    elif backend in ("alchemi", "alchemi_naive"):
        edge_index, cell_offsets = build_neighbor_list_batch_alchemi(
            positions=positions, cell=cell, pbc=pbc, batch=batch, cutoff=cutoff,
            method="naive" if backend == "alchemi_naive" else "cell_list",
        )
    else:
        raise ValueError(f"Unsupported PBC neighbor backend: {backend}")
    return _limit_neighbors(
        edge_index=edge_index,
        cell_offsets=cell_offsets,
        positions=positions,
        cell=cell,
        batch=batch,
        max_num_neighbors=max_num_neighbors,
    )


def expand_ghost_nodes(
    positions: Tensor,
    z: Tensor,
    cell: Tensor,
    pbc: Tensor,
    batch: Tensor,
    cutoff: float,
    backend: str = "torch",
    max_num_neighbors: int | None = None,
) -> ExpandedPBCGraph:
    edge_index, cell_offsets = build_neighbor_list_batch(
        positions=positions,
        cell=cell,
        pbc=pbc,
        batch=batch,
        cutoff=cutoff,
        backend=backend,
        max_num_neighbors=max_num_neighbors,
    )
    batch_size, cell, _ = _normalize_batch_inputs(cell, pbc, batch)
    if batch_size == 0 or edge_index.numel() == 0:
        return ExpandedPBCGraph(
            expanded_pos=positions,
            expanded_z=z,
            expanded_batch=batch,
            expanded_edge_index=edge_index,
            num_real_nodes=int(positions.shape[0]),
            ghost2real_index=torch.empty((0,), dtype=torch.long, device=positions.device),
        )

    src, dst = edge_index
    cross_mask = (cell_offsets != 0).any(dim=1)
    if not torch.any(cross_mask):
        return ExpandedPBCGraph(
            expanded_pos=positions,
            expanded_z=z,
            expanded_batch=batch,
            expanded_edge_index=edge_index,
            num_real_nodes=int(positions.shape[0]),
            ghost2real_index=torch.empty((0,), dtype=torch.long, device=positions.device),
        )

    cross_src = src[cross_mask]
    cross_offsets = cell_offsets[cross_mask]
    cross_batch = batch.index_select(0, cross_src)

    ghost_keys = torch.cat([cross_batch.unsqueeze(1), cross_src.unsqueeze(1), cross_offsets], dim=1)
    unique_keys, ghost_inverse = torch.unique(ghost_keys, dim=0, sorted=True, return_inverse=True)
    unique_batch = unique_keys[:, 0]
    unique_src = unique_keys[:, 1]
    unique_offsets = unique_keys[:, 2:]

    shift_vec = _cell_shifts_to_cartesian(unique_offsets, cell, unique_batch, positions.dtype)
    ghost_pos = positions.index_select(0, unique_src) - shift_vec
    ghost_z = z.index_select(0, unique_src)

    num_real_nodes = int(positions.shape[0])
    ghost_node_index = torch.arange(unique_src.shape[0], device=positions.device, dtype=torch.long) + num_real_nodes

    expanded_pos = torch.cat([positions, ghost_pos], dim=0)
    expanded_z = torch.cat([z, ghost_z], dim=0)
    expanded_batch = torch.cat([batch, unique_batch], dim=0)

    expanded_src = src.clone()
    expanded_src[cross_mask] = ghost_node_index.index_select(0, ghost_inverse)
    expanded_edge_index = torch.stack([expanded_src, dst], dim=0)
    if expanded_edge_index.numel() > 0:
        expanded_edge_index = torch.unique(expanded_edge_index.t(), dim=0, sorted=True).t().contiguous()

    return ExpandedPBCGraph(
        expanded_pos=expanded_pos,
        expanded_z=expanded_z,
        expanded_batch=expanded_batch,
        expanded_edge_index=expanded_edge_index,
        num_real_nodes=num_real_nodes,
        ghost2real_index=unique_src,
    )
