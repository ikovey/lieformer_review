"""Exact radius-limited top-k PBC neighbors using a Warp branch-and-bound query.

No radius graph is materialized. A fractional-coordinate cell grid indexes the
real atoms once; periodic images are visited implicitly. Each query maintains a
max heap of k candidates and prunes cells using conservative distance bounds.
This optional backend requires warp-lang, but does not depend on nvalchemi.
"""

from functools import lru_cache
import math

import torch


@lru_cache(maxsize=16)
def _query_kernel(dtype):
    import warp as wp

    scalar = wp.float64 if dtype == torch.float64 else wp.float32
    vec = wp.types.vector(length=3, dtype=scalar)
    mat = wp.types.matrix(shape=(3, 3), dtype=scalar)

    @wp.func
    def worse(d1: scalar, a1: int, s1: wp.vec3i, d2: scalar, a2: int, s2: wp.vec3i):
        # Deterministic lexicographic ordering for exact distance ties.
        return d1 > d2 or (d1 == d2 and (a1 > a2 or (a1 == a2 and (
            s1[0] > s2[0] or (s1[0] == s2[0] and (
                s1[1] > s2[1] or (s1[1] == s2[1] and s1[2] > s2[2])))))))

    @wp.kernel(enable_backward=False)
    def query(
        pos: wp.array(dtype=vec), frac: wp.array(dtype=vec),
        wraps: wp.array(dtype=wp.vec3i), batch: wp.array(dtype=wp.int32),
        cell: wp.array(dtype=mat), periodic: wp.array(dtype=wp.vec3i),
        origin: wp.array(dtype=vec), width: wp.array(dtype=vec),
        heights: wp.array(dtype=vec), dims: wp.array(dtype=wp.vec3i),
        grid_offset: wp.array(dtype=wp.int32), atom_bin: wp.array(dtype=wp.vec3i),
        starts: wp.array(dtype=wp.int32), ends: wp.array(dtype=wp.int32),
        ordered_atoms: wp.array(dtype=wp.int32), cutoff: scalar, k: int,
        indices: wp.array2d(dtype=wp.int32), shifts: wp.array2d(dtype=wp.vec3i),
        distances: wp.array2d(dtype=scalar), counts: wp.array(dtype=wp.int32),
        statistics: wp.array(dtype=wp.vec3i),
    ):
        target = wp.tid()
        graph = batch[target]
        f = frac[target]
        c = cell[graph]
        shape = dims[graph]
        center = atom_bin[target]
        step = width[graph]
        org = origin[graph]
        height = heights[graph]
        pbc = periodic[graph]
        count = int(0)
        ring = int(0)
        tested = int(0)
        visited = int(0)
        pruned = int(0)
        limit2 = cutoff * cutoff
        # Relax lower bounds to avoid pruning on fractional-coordinate roundoff.
        geometry_scale = cutoff
        for a in range(3):
            for b in range(3):
                geometry_scale = wp.max(geometry_scale, wp.abs(c[a, b]))
        margin = scalar(0.00001) * geometry_scale
        while True:
            for dx in range(-ring, ring + 1):
                for dy in range(-ring, ring + 1):
                    for dz in range(-ring, ring + 1):
                        if wp.max(wp.abs(dx), wp.max(wp.abs(dy), wp.abs(dz))) != ring:
                            continue
                        ijk = center + wp.vec3i(dx, dy, dz)
                        valid = bool(True)
                        image = wp.vec3i(0)
                        wrapped = wp.vec3i(0)
                        bound = scalar(0)
                        for axis in range(3):
                            if pbc[axis] == 0 and (ijk[axis] < 0 or ijk[axis] >= shape[axis]):
                                valid = False
                            image[axis] = int(wp.floor(scalar(ijk[axis]) / scalar(shape[axis])))
                            wrapped[axis] = ijk[axis] - image[axis] * shape[axis]
                            lo = org[axis] + scalar(ijk[axis]) * step[axis]
                            hi = lo + step[axis]
                            gap = wp.max(scalar(0), wp.max(lo - f[axis], f[axis] - hi))
                            bound = wp.max(bound, gap * height[axis])
                        if not valid:
                            continue
                        relaxed_bound = wp.max(scalar(0), bound - margin)
                        if relaxed_bound * relaxed_bound > limit2:
                            pruned += 1
                            continue
                        key = grid_offset[graph] + (wrapped[0] * shape[1] + wrapped[1]) * shape[2] + wrapped[2]
                        visited += 1
                        for cursor in range(starts[key], ends[key]):
                            source = ordered_atoms[cursor]
                            if source == target and image == wp.vec3i(0):
                                continue
                            tested += 1
                            # Return offsets for edge source -> target, matching
                            # pos[target] - pos[source] + offset @ cell.
                            shift = wraps[source] - wraps[target] - image
                            shift_cart = vec(scalar(shift[0]), scalar(shift[1]), scalar(shift[2])) * c
                            delta = pos[target] - pos[source] + shift_cart
                            d2 = wp.dot(delta, delta)
                            if d2 >= cutoff * cutoff:
                                continue
                            if count < k:
                                at = count
                                count += 1
                                while at > 0:
                                    parent = (at - 1) // 2
                                    if not worse(d2, source, shift, distances[target, parent], indices[target, parent], shifts[target, parent]):
                                        break
                                    distances[target, at] = distances[target, parent]
                                    indices[target, at] = indices[target, parent]
                                    shifts[target, at] = shifts[target, parent]
                                    at = parent
                                distances[target, at] = d2
                                indices[target, at] = source
                                shifts[target, at] = shift
                            elif worse(distances[target, 0], indices[target, 0], shifts[target, 0], d2, source, shift):
                                at = int(0)
                                while 2 * at + 1 < k:
                                    child = 2 * at + 1
                                    right = child + 1
                                    if right < k and worse(distances[target, right], indices[target, right], shifts[target, right], distances[target, child], indices[target, child], shifts[target, child]):
                                        child = right
                                    if not worse(distances[target, child], indices[target, child], shifts[target, child], d2, source, shift):
                                        break
                                    distances[target, at] = distances[target, child]
                                    indices[target, at] = indices[target, child]
                                    shifts[target, at] = shifts[target, child]
                                    at = child
                                distances[target, at] = d2
                                indices[target, at] = source
                                shifts[target, at] = shift
                            if count == k:
                                limit2 = distances[target, 0]
            # Everything outside the visited bin box crosses at least one of
            # its six planes. Fractional reciprocal-plane distances are valid
            # lower bounds even for a triclinic cell (no minimum-image shortcut).
            remaining = scalar(1.0e30)
            for axis in range(3):
                lower_bin = center[axis] - ring
                upper_bin = center[axis] + ring + 1
                if pbc[axis] != 0 or lower_bin > 0:
                    gap = (f[axis] - (org[axis] + scalar(lower_bin) * step[axis])) * height[axis]
                    remaining = wp.min(remaining, gap)
                if pbc[axis] != 0 or upper_bin < shape[axis]:
                    gap = (org[axis] + scalar(upper_bin) * step[axis] - f[axis]) * height[axis]
                    remaining = wp.min(remaining, gap)
            relaxed_remaining = wp.max(scalar(0), remaining - margin)
            if relaxed_remaining * relaxed_remaining > limit2:
                break
            ring += 1
        counts[target] = count
        statistics[target] = wp.vec3i(tested, visited, pruned)

    return query


@torch.no_grad()
def build_neighbor_list_batch_warp(
    positions, cell, pbc, batch, cutoff, max_num_neighbors,
    *, bin_width=None, return_statistics=False,
):
    """Return incoming top-k edges inside cutoff; ties use (source, offset).

    Geometry is detached, as for existing graph builders. The model computes
    differentiable edge vectors from the returned indices and integer shifts.
    """
    import warp as wp

    if positions.device.type != "cuda" or positions.dtype not in (torch.float32, torch.float64):
        raise ValueError("Warp top-k requires CUDA float32 or float64 positions")
    if max_num_neighbors is None or max_num_neighbors <= 0 or not math.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("Warp top-k requires a positive neighbor cap and cutoff")
    n = positions.shape[0]
    if n == 0:
        result = (torch.empty(2, 0, dtype=torch.long, device=positions.device),
                  torch.empty(0, 3, dtype=torch.long, device=positions.device))
        return (*result, {}) if return_statistics else result
    # CUDA from_torch accesses Warp's device table directly. Do not rely on
    # another package/import having initialized it first (fresh worker startup).
    # wp.init() is idempotent; subsequent graph builds reuse the runtime.
    wp.init()
    device, dtype = positions.device, positions.dtype
    batch = batch.to(device=device, dtype=torch.long)
    graphs = int(batch.max()) + 1
    if cell.ndim == 2:
        cell = cell[None]
    if pbc.ndim == 1:
        pbc = pbc[None]
    cell = cell.expand(graphs, -1, -1) if cell.shape[0] == 1 else cell[:graphs]
    pbc = pbc.expand(graphs, -1) if pbc.shape[0] == 1 else pbc[:graphs]
    if cell.shape[0] != graphs or pbc.shape[0] != graphs:
        raise ValueError("Missing cell or PBC metadata for a batch graph")
    cell = cell.to(dtype=dtype, device=device).contiguous()
    periodic = pbc.to(device=device, dtype=torch.bool)
    inv = torch.linalg.inv(cell)
    height = inv.norm(dim=1).reciprocal()
    # Geometry bounds must not inherit the model's TF32 matmul approximation.
    frac = (positions[:, :, None] * inv[batch]).sum(dim=1)
    if not bool(torch.isfinite(frac).all() & torch.isfinite(height).all() & (height > 0).all()):
        raise ValueError("Warp top-k requires finite positions and nonsingular finite cells")
    wraps = torch.where(periodic[batch], frac.floor(), 0).int()
    frac = frac - wraps
    expanded_batch = batch[:, None].expand(-1, 3)
    lo = torch.full((graphs, 3), float("inf"), device=device, dtype=dtype)
    hi = -lo
    lo.scatter_reduce_(0, expanded_batch, frac, reduce="amin", include_self=True)
    hi.scatter_reduce_(0, expanded_batch, frac, reduce="amax", include_self=True)
    occupied = torch.bincount(batch, minlength=graphs)
    lo = torch.where(occupied[:, None] > 0, lo, 0)
    hi = torch.where(occupied[:, None] > 0, hi, 0)
    origin = torch.where(periodic, 0, lo - 1e-5)
    extent = torch.where(periodic, 1, hi - lo + 2e-5)
    bin_width = float(bin_width if bin_width is not None else cutoff / 4)
    if not math.isfinite(bin_width) or bin_width <= 0:
        raise ValueError("bin_width must be positive")
    dims = (extent * height / bin_width).floor().clamp_min(1).long()
    # Bound dense grid metadata on elongated or mostly empty cells.
    axis_cap = (occupied * 8 + 64).double().pow(1 / 3).ceil().long()
    dims = torch.minimum(dims, axis_cap[:, None])
    width = extent / dims
    atom_bin = ((frac - origin[batch]) / width[batch]).floor().long()
    atom_bin = torch.minimum(atom_bin.clamp_min(0), dims[batch] - 1)
    grid_sizes = dims.prod(dim=1)
    offsets = grid_sizes.cumsum(0) - grid_sizes
    total_cells = int(grid_sizes.sum())
    keys = offsets[batch] + (atom_bin[:, 0] * dims[batch, 1] + atom_bin[:, 1]) * dims[batch, 2] + atom_bin[:, 2]
    order = torch.argsort(keys, stable=True).int()
    bin_counts = torch.bincount(keys, minlength=total_cells)
    ends = bin_counts.cumsum(0).int()
    starts = ends - bin_counts.int()
    k = int(max_num_neighbors)
    indices = torch.empty((n, k), device=device, dtype=torch.int32)
    shifts = torch.empty((n, k, 3), device=device, dtype=torch.int32)
    distances = torch.empty((n, k), device=device, dtype=dtype)
    counts = torch.empty(n, device=device, dtype=torch.int32)
    stats = torch.empty(n, 3, device=device, dtype=torch.int32)
    scalar = wp.float64 if dtype == torch.float64 else wp.float32
    vec = wp.types.vector(length=3, dtype=scalar)
    mat = wp.types.matrix(shape=(3, 3), dtype=scalar)
    arrays = [
        (positions.contiguous(), vec), (frac.contiguous(), vec), (wraps.contiguous(), wp.vec3i),
        (batch.int(), wp.int32), (cell, mat), (periodic.int(), wp.vec3i),
        (origin.contiguous(), vec), (width.contiguous(), vec), (height.contiguous(), vec),
        (dims.int(), wp.vec3i), (offsets.int(), wp.int32), (atom_bin.int(), wp.vec3i),
        (starts, wp.int32), (ends, wp.int32), (order, wp.int32),
    ]
    wp.launch(_query_kernel(dtype), dim=n,
        inputs=[wp.from_torch(t, dtype=w) for t, w in arrays] + [cutoff, k],
        outputs=[wp.from_torch(indices), wp.from_torch(shifts, dtype=wp.vec3i),
                 wp.from_torch(distances), wp.from_torch(counts), wp.from_torch(stats, dtype=wp.vec3i)],
        stream=wp.stream_from_torch(torch.cuda.current_stream(device)), block_dim=64)
    row, slot = torch.nonzero(torch.arange(k, device=device)[None] < counts[:, None], as_tuple=True)
    edge_index = torch.stack([indices[row, slot].long(), row])
    cell_offsets = shifts[row, slot].long()
    if return_statistics:
        return edge_index, cell_offsets, dict(
            tested_candidates=stats[:, 0], visited_bins=stats[:, 1], pruned_bins=stats[:, 2],
            grid_cells=total_cells,
        )
    return edge_index, cell_offsets
