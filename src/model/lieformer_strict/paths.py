"""Outer full-admissible paths and their exact Wigner-6j expansion."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

from sympy.physics.wigner import wigner_6j

from .deps.path_pruning import OuterPathPolicy, build_outer_path_specs


@dataclass(frozen=True)
class OuterPath:
    path_id: int
    input_degree: int
    geometry_degree: int
    output_degree: int


@dataclass(frozen=True)
class RecoupledPath:
    outer_path_id: int
    input_degree: int
    geometry_degree: int
    output_degree: int
    source_geometry_degree: int
    destination_geometry_degree: int
    intermediate_degree: int
    coefficient: float


def normalize_geometry_orders(geometry_orders: Iterable[int]) -> tuple[int, ...]:
    orders = tuple(sorted(set(int(order) for order in geometry_orders)))
    if not orders or orders[0] < 0:
        raise ValueError(f"geometry_orders must be non-empty and non-negative, got {orders}")
    return orders


def build_full_admissible_paths(
    feature_lmax: int,
    geometry_orders: Iterable[int] = (0, 1, 2),
) -> tuple[OuterPath, ...]:
    """Enumerate every admissible ``(input, geometry, output)`` outer path."""
    if feature_lmax < 0:
        raise ValueError(f"feature_lmax must be non-negative, got {feature_lmax}")
    orders = normalize_geometry_orders(geometry_orders)
    return build_outer_paths(feature_lmax, orders, "full_admissible")


def build_outer_paths(
    feature_lmax: int,
    geometry_orders: Iterable[int],
    outer_path_policy: str | dict | OuterPathPolicy,
) -> tuple[OuterPath, ...]:
    """Select outer paths through the isolated policy infrastructure."""

    orders = normalize_geometry_orders(geometry_orders)
    specs = build_outer_path_specs(feature_lmax, orders, outer_path_policy)
    return tuple(
        OuterPath(
            path_id=path_id,
            input_degree=spec.input_degree,
            geometry_degree=spec.geometry_degree,
            output_degree=spec.output_degree,
        )
        for path_id, spec in enumerate(specs)
    )


@lru_cache(maxsize=None)
def _recoupling_coefficient(a: int, n: int, output_degree: int, b: int, d: int) -> float:
    c = n - b
    six_j = wigner_6j(a, b, d, c, output_degree, n)
    if six_j == 0:
        return 0.0
    return float(
        math.comb(n, b)
        * ((-1) ** b)
        * ((-1) ** (a + b + c + output_degree))
        * math.sqrt((2 * d + 1) * (2 * n + 1))
        * six_j
    )


def recouple_outer_path(path: OuterPath) -> tuple[RecoupledPath, ...]:
    a = path.input_degree
    n = path.geometry_degree
    output_degree = path.output_degree
    branches: list[RecoupledPath] = []
    for b in range(n + 1):
        c = n - b
        d_min = max(abs(a - b), abs(output_degree - c))
        d_max = min(a + b, output_degree + c)
        for d in range(d_min, d_max + 1):
            coefficient = _recoupling_coefficient(a, n, output_degree, b, d)
            if coefficient == 0.0:
                continue
            branches.append(
                RecoupledPath(
                    outer_path_id=path.path_id,
                    input_degree=a,
                    geometry_degree=n,
                    output_degree=output_degree,
                    source_geometry_degree=b,
                    destination_geometry_degree=c,
                    intermediate_degree=d,
                    coefficient=coefficient,
                )
            )
    return tuple(branches)


def recouple_paths(paths: Iterable[OuterPath]) -> dict[int, tuple[RecoupledPath, ...]]:
    return {path.path_id: recouple_outer_path(path) for path in paths}
