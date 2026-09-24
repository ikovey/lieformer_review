"""Self-contained SO(3) tensor-product and solid-harmonic primitives for SHR."""
from __future__ import annotations

import math
from collections.abc import Iterable
from functools import lru_cache

import torch
from e3nn import o3
from torch import Tensor


@lru_cache(maxsize=None)
def _cached_wigner_3j(l1: int, l2: int, lout: int, dtype: torch.dtype, device_type: str, device_index: int | None) -> Tensor:
    return o3.wigner_3j(l1, l2, lout, dtype=dtype, device=torch.device(device_type, device_index))


@lru_cache(maxsize=None)
def _cached_geometry_coupling(l1, l2, lout, dtype, device_type, device_index):
    w = _cached_wigner_3j(l1, l2, lout, dtype, device_type, device_index)
    return (math.sqrt(2 * lout + 1) * w.permute(1, 2, 0).contiguous()).reshape(2 * l2 + 1, -1)


def component_tensor_product(x: Tensor, y: Tensor, l1: int, l2: int, lout: int) -> Tensor:
    """Couple ``x[..., 2*l1+1, channels]`` and ``y[..., 2*l2+1]`` to ``lout``."""
    if x.shape[-2] != 2 * l1 + 1:
        raise ValueError(f"x degree mismatch: expected {2*l1+1}, got {tuple(x.shape)}")
    if y.shape[-1] != 2 * l2 + 1:
        raise ValueError(f"y degree mismatch: expected {2*l2+1}, got {tuple(y.shape)}")
    if not abs(l1 - l2) <= lout <= l1 + l2:
        raise ValueError(f"inadmissible tensor product ({l1}, {l2}) -> {lout}")
    # Contract the small geometry/CG axes before introducing feature channels.
    # This fixes the contraction plan and avoids opt_einsum path search in every
    # forward, as well as channel-wide outer products and a final scaling pass.
    coupling = _cached_geometry_coupling(l1, l2, lout, x.dtype, x.device.type, x.device.index)
    y = y.to(dtype=x.dtype, device=x.device)
    matrix = (y @ coupling).reshape(y.shape[:-1] + (2 * lout + 1, 2 * l1 + 1))
    return torch.matmul(matrix, x)


def scatter_sum(source: Tensor, index: Tensor, dim_size: int) -> Tensor:
    if index.ndim != 1 or index.shape[0] != source.shape[0]:
        raise ValueError("index must have shape (E,) and align with source")
    if index.dtype != torch.long:
        raise TypeError("index must be torch.long")
    return source.new_zeros((dim_size,) + source.shape[1:]).index_add_(0, index, source)


def solid_harmonic_scale(degree: int) -> float:
    if degree < 0:
        raise ValueError("degree must be non-negative")
    return math.sqrt(4.0 * math.pi * (2**degree) * math.factorial(degree) ** 2 / math.factorial(2 * degree + 1))


def regular_solid_harmonics(vectors: Tensor, degrees: int | Iterable[int]) -> dict[int, Tensor]:
    if vectors.ndim < 1 or vectors.shape[-1] != 3:
        raise ValueError(f"vectors must end in 3 Cartesian components, got {tuple(vectors.shape)}")
    degree_list = tuple(range(degrees + 1)) if isinstance(degrees, int) else tuple(sorted(set(int(x) for x in degrees)))
    if not degree_list or degree_list[0] < 0:
        raise ValueError("degrees must be a non-empty collection of non-negative integers")
    out: dict[int, Tensor] = {}
    for degree in degree_list:
        if degree == 0:
            out[degree] = vectors.new_ones(vectors.shape[:-1] + (1,))
        else:
            out[degree] = solid_harmonic_scale(degree) * o3.spherical_harmonics(
                degree, vectors, normalize=False, normalization="integral"
            )
    return out


def center_positions(positions: Tensor, batch: Tensor | None = None) -> Tensor:
    if positions.ndim != 2 or positions.shape[-1] != 3:
        raise ValueError(f"positions must have shape (N, 3), got {tuple(positions.shape)}")
    if batch is None:
        return positions - positions.mean(dim=0, keepdim=True) if positions.shape[0] else positions
    if batch.ndim != 1 or batch.shape[0] != positions.shape[0] or batch.dtype != torch.long:
        raise ValueError("batch must be a long tensor with one entry per position")
    if not batch.numel():
        return positions
    ngraphs = int(batch.max().item()) + 1
    centers = positions.new_zeros((ngraphs, 3)).index_add_(0, batch, positions)
    counts = positions.new_zeros((ngraphs, 1)).index_add_(0, batch, positions.new_ones((positions.shape[0], 1)))
    return positions - (centers / counts.clamp_min(1.0)).index_select(0, batch)
