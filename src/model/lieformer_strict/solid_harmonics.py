"""Regular solid harmonics in the convention used by the strict TP."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from e3nn import o3
from torch import Tensor


def solid_harmonic_scale(degree: int) -> float:
    """Scale e3nn integral-normalized harmonics into a binomial translation basis."""
    if degree < 0:
        raise ValueError(f"degree must be non-negative, got {degree}")
    return math.sqrt(
        4.0
        * math.pi
        * (2**degree)
        * (math.factorial(degree) ** 2)
        / math.factorial(2 * degree + 1)
    )


def regular_solid_harmonics(
    vectors: Tensor,
    degrees: int | Iterable[int],
) -> dict[int, Tensor]:
    """Return degree-separated regular solid harmonics.

    The returned tensors have shape ``vectors.shape[:-1] + (2*l+1,)`` and obey
    the finite binomial translation identity used by the recoupling code.
    """
    if vectors.shape[-1] != 3:
        raise ValueError(f"vectors must end in a Cartesian dimension of 3, got {tuple(vectors.shape)}")
    if isinstance(degrees, int):
        degree_list = tuple(range(degrees + 1))
    else:
        degree_list = tuple(sorted(set(int(degree) for degree in degrees)))
    if not degree_list or degree_list[0] < 0:
        raise ValueError(f"degrees must be a non-empty collection of non-negative integers, got {degree_list}")

    result: dict[int, Tensor] = {}
    for degree in degree_list:
        if degree == 0:
            result[degree] = vectors.new_ones(vectors.shape[:-1] + (1,))
        else:
            result[degree] = solid_harmonic_scale(degree) * o3.spherical_harmonics(
                degree,
                vectors,
                normalize=False,
                normalization="integral",
            )
    return result


def center_positions(positions: Tensor, batch: Tensor | None = None) -> Tensor:
    """Subtract one common origin per graph without changing relative vectors."""
    if positions.ndim != 2 or positions.shape[-1] != 3:
        raise ValueError(f"positions must have shape (N, 3), got {tuple(positions.shape)}")
    if positions.shape[0] == 0:
        return positions
    if batch is None:
        return positions - positions.mean(dim=0, keepdim=True)
    if batch.ndim != 1 or batch.shape[0] != positions.shape[0]:
        raise ValueError("batch must have shape (N,) and align with positions")
    if batch.dtype != torch.long:
        raise TypeError(f"batch must use torch.long indices, got {batch.dtype}")
    num_graphs = int(batch.max().item()) + 1
    centers = positions.new_zeros((num_graphs, 3))
    centers.index_add_(0, batch, positions)
    counts = positions.new_zeros((num_graphs, 1))
    counts.index_add_(0, batch, positions.new_ones((positions.shape[0], 1)))
    centers = centers / counts.clamp_min(1.0)
    return positions - centers.index_select(0, batch)
