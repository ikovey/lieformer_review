"""Small component-normalized SO(3) tensor-product primitives."""

from __future__ import annotations

import math
from functools import lru_cache

import torch
from e3nn import o3
from torch import Tensor


@lru_cache(maxsize=None)
def _cached_wigner_3j(
    l1: int,
    l2: int,
    lout: int,
    dtype: torch.dtype,
    device_type: str,
    device_index: int | None,
) -> Tensor:
    device = torch.device(device_type, device_index)
    return o3.wigner_3j(l1, l2, lout, dtype=dtype, device=device)


def component_tensor_product(x: Tensor, y: Tensor, l1: int, l2: int, lout: int) -> Tensor:
    """Couple ``x[..., m1, channel]`` and ``y[..., m2]`` into degree ``lout``."""
    if x.shape[-2] != 2 * l1 + 1:
        raise ValueError(f"x degree mismatch: l1={l1}, shape={tuple(x.shape)}")
    if y.shape[-1] != 2 * l2 + 1:
        raise ValueError(f"y degree mismatch: l2={l2}, shape={tuple(y.shape)}")
    if not abs(l1 - l2) <= lout <= l1 + l2:
        raise ValueError(f"inadmissible tensor product ({l1}, {l2}) -> {lout}")
    w3j = _cached_wigner_3j(
        l1,
        l2,
        lout,
        x.dtype,
        x.device.type,
        x.device.index,
    )
    y = y.to(dtype=x.dtype, device=x.device)
    return math.sqrt(2 * lout + 1) * torch.einsum("...ic,...j,ijk->...kc", x, y, w3j)


def scatter_sum(source: Tensor, index: Tensor, dim_size: int) -> Tensor:
    """Self-contained differentiable scatter sum along the leading dimension."""
    if index.ndim != 1 or index.shape[0] != source.shape[0]:
        raise ValueError("index must have shape (E,) and align with source")
    output = source.new_zeros((dim_size,) + source.shape[1:])
    output.index_add_(0, index, source)
    return output
