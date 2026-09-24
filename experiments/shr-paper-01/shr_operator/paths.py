"""Outer CG paths and Wigner-6j recoupling tables for Sparse Harmonic Recoupling."""
from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache

from sympy.physics.wigner import wigner_6j


@dataclass(frozen=True, order=True)
class OuterPath:
    path_id: int
    input_degree: int
    geometry_degree: int
    output_degree: int

    @property
    def l_in(self) -> int: return self.input_degree
    @property
    def ell(self) -> int: return self.geometry_degree
    @property
    def input_l(self) -> int: return self.input_degree
    @property
    def output_l(self) -> int: return self.output_degree
    @property
    def l_out(self) -> int: return self.output_degree


@dataclass(frozen=True, order=True)
class SHRBranch:
    """One nonzero recoupled branch indexed by source order ``a`` and ``f``."""
    outer_path_id: int
    input_degree: int
    geometry_degree: int
    output_degree: int
    source_degree: int
    target_degree: int
    intermediate_degree: int
    coefficient: float

    @property
    def a(self) -> int: return self.source_degree
    @property
    def u(self) -> int: return self.target_degree
    @property
    def f(self) -> int: return self.intermediate_degree
    @property
    def ell(self) -> int: return self.geometry_degree
    @property
    def source_geometry_degree(self) -> int: return self.source_degree
    @property
    def destination_geometry_degree(self) -> int: return self.target_degree
    @property
    def input_l(self) -> int: return self.input_degree
    @property
    def output_l(self) -> int: return self.output_degree


def _admissible(a: int, b: int, c: int) -> bool:
    return abs(a - b) <= c <= a + b


def all_admissible_paths(
    input_degrees: Iterable[int] | int | None = None,
    output_degrees: Iterable[int] | int | None = None,
    geometry_orders: Iterable[int] = (0, 1, 2),
    *,
    feature_lmax: int | None = None,
    input_lmax: int | None = None,
    output_lmax: int | None = None,
) -> tuple[tuple[int, int, int], ...]:
    if feature_lmax is not None:
        input_lmax = feature_lmax if input_lmax is None else input_lmax
        output_lmax = feature_lmax if output_lmax is None else output_lmax
    if input_degrees is None:
        if input_lmax is None: raise ValueError("input_degrees or input_lmax is required")
        input_degrees = range(int(input_lmax) + 1)
    elif not isinstance(input_degrees, int):
        input_degrees = tuple(input_degrees)
    if output_degrees is None:
        if output_lmax is None: output_lmax = int(input_lmax if input_lmax is not None else max(input_degrees))
        output_degrees = range(int(output_lmax) + 1)
    if isinstance(input_degrees, int): input_degrees = range(input_degrees + 1)
    if isinstance(output_degrees, int): output_degrees = range(output_degrees + 1)
    ins = tuple(sorted(set(int(x) for x in input_degrees)))
    outs = tuple(sorted(set(int(x) for x in output_degrees)))
    geos = tuple(sorted(set(int(x) for x in geometry_orders)))
    if not ins or not outs or not geos or min(ins + outs + geos) < 0:
        raise ValueError("degrees and geometry_orders must be non-empty and non-negative")
    return tuple((l, ell, q) for l in ins for ell in geos for q in outs if _admissible(l, ell, q))


def build_outer_paths(
    input_degrees: Iterable[int] | int | None = None,
    output_degrees: Iterable[int] | int | None = None,
    geometry_orders: Iterable[int] = (0, 1, 2),
    *,
    feature_lmax: int | None = None,
    input_lmax: int | None = None,
    output_lmax: int | None = None,
    K: int | None = None,
    outer_path_subset: Iterable[tuple[int, int, int]] | None = None,
) -> tuple[OuterPath, ...]:
    """Build admissible ``(l_in, ell, l_out)`` paths.

    ``K`` is an optional outer-path sparsity control. It caps the number of
    canonical paths after admissibility filtering; it never changes the inner
    recoupled bandwidth. ``outer_path_subset`` can be used for an explicit
    ``P_K`` set and takes precedence over ``K``.
    """
    if feature_lmax is not None:
        input_lmax = feature_lmax if input_lmax is None else input_lmax
        output_lmax = feature_lmax if output_lmax is None else output_lmax
    if input_degrees is None:
        if input_lmax is None: raise ValueError("input_degrees or input_lmax is required")
        input_degrees = range(int(input_lmax) + 1)
    if output_degrees is None and output_lmax is not None:
        output_degrees = range(int(output_lmax) + 1)
    candidates = all_admissible_paths(input_degrees, output_degrees, geometry_orders, input_lmax=input_lmax, output_lmax=output_lmax)
    if outer_path_subset is not None:
        selected = set(tuple(map(int, p)) for p in outer_path_subset)
        unknown = selected.difference(candidates)
        if unknown:
            raise ValueError(f"outer_path_subset contains inadmissible paths: {sorted(unknown)}")
        candidates = tuple(path for path in candidates if path in selected)
    elif K is not None:
        if int(K) < 0:
            raise ValueError("K must be non-negative")
        candidates = candidates[: int(K)]
    return tuple(OuterPath(i, *path) for i, path in enumerate(candidates))


@lru_cache(maxsize=None)
def recoupling_coefficient(l_in: int, ell: int, l_out: int, source_degree: int, f: int) -> float:
    """Exact translation-plus-6j coefficient for ``a=source_degree``.

    The convention matches ``regular_solid_harmonics`` and the component TP:
    ``binom(ell,a) (-1)^a (-1)^(l_in+a+u+l_out)
    sqrt((2f+1)(2ell+1)) {l_in a f; u l_out ell}``, with ``u=ell-a``.
    """
    a = int(source_degree)
    u = int(ell - a)
    if a < 0 or u < 0 or not _admissible(l_in, a, f) or not _admissible(f, u, l_out):
        return 0.0
    six = wigner_6j(l_in, a, f, u, l_out, ell)
    if six == 0:
        return 0.0
    return float(
        math.comb(ell, a)
        * ((-1) ** a)
        * ((-1) ** (l_in + a + u + l_out))
        * math.sqrt((2 * f + 1) * (2 * ell + 1))
        * six
    )


def recouple_path(path: OuterPath, B: int | None = None, *, bandwidth: int | None = None) -> tuple[SHRBranch, ...]:
    if bandwidth is not None: B = bandwidth if B is None else B
    if B is not None and int(B) < 0:
        raise ValueError("B must be non-negative or None")
    branches: list[SHRBranch] = []
    l, ell, q = path.input_degree, path.geometry_degree, path.output_degree
    for a in range(ell + 1):
        u = ell - a
        fmin, fmax = max(abs(l - a), abs(q - u)), min(l + a, q + u)
        for f in range(fmin, fmax + 1):
            if B is not None and abs(f - q) > int(B):
                continue
            coeff = recoupling_coefficient(l, ell, q, a, f)
            if coeff:
                branches.append(SHRBranch(path.path_id, l, ell, q, a, u, f, coeff))
    return tuple(branches)


def recouple_paths(paths: Iterable[OuterPath], B: int | None = None, *, bandwidth: int | None = None) -> dict[int, tuple[SHRBranch, ...]]:
    if bandwidth is not None: B = bandwidth if B is None else B
    return {path.path_id: recouple_path(path, B=B) for path in paths}


# Name used by the earlier strict implementation; retained as a harmless alias.
RecoupledPath = SHRBranch
