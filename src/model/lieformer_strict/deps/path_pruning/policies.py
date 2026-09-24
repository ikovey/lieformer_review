"""Built-in outer-path policies.

Only the outer triplet set is pruned here.  Every retained triplet still gets
its complete Wigner-6j expansion in ``lieformer_strict.paths``.
"""

from __future__ import annotations

from dataclasses import dataclass

from .core import AngularPath, PathSelectionContext, path_policy_registry


@dataclass(frozen=True)
class FullAdmissiblePolicy:
    name: str = "full_admissible"

    def select(self, candidates, context: PathSelectionContext):
        del context
        return candidates


@dataclass(frozen=True)
class ZippedPolicy:
    """The distinct outer-triplet family used by the original ZiTP.

    ``include_zero_frequency`` controls the old ``use_zfp`` family ``(i,i,0)``.
    Duplicate raw instructions (notably ``(1,1,0)``) are deliberately merged:
    this policy defines a mathematical subset of P1, not an instruction list.
    """

    include_zero_frequency: bool = True
    name: str = "zipped"

    def select(self, candidates, context: PathSelectionContext):
        zipped: set[AngularPath] = set()
        for degree in range(1, context.feature_lmax + 1):
            if self.include_zero_frequency:
                zipped.add(AngularPath(degree, degree, 0))
            if degree % 2 == 0:
                zipped.update(
                    {
                        AngularPath(degree, degree, degree),
                        AngularPath(degree, degree - 1, degree - 1),
                        AngularPath(degree - 1, degree, degree - 1),
                    }
                )
            else:
                zipped.update(
                    {
                        AngularPath(degree, degree - 1, degree),
                        AngularPath(degree - 1, degree, degree),
                        AngularPath(degree, degree, degree - 1),
                    }
                )
                if degree < context.feature_lmax:
                    zipped.add(AngularPath(degree, degree, degree + 1))
        return (path for path in candidates if path in zipped)


@dataclass(frozen=True)
class ExplicitPathPolicy:
    """Select an explicitly recorded set of angular triplets.

    This is the static deployment format for searched V3 architectures.  The
    generic subset validation in :func:`build_outer_path_specs` remains the
    authority for admissibility and canonical ordering.
    """

    paths: tuple[tuple[int, int, int], ...] | list[list[int]]
    name: str = "explicit"

    def __post_init__(self) -> None:
        normalized: list[tuple[int, int, int]] = []
        for index, raw_path in enumerate(self.paths):
            if not isinstance(raw_path, (list, tuple)) or len(raw_path) != 3:
                raise ValueError(
                    f"explicit path {index} must be an [input, geometry, output] triplet"
                )
            if any(isinstance(value, bool) or not isinstance(value, int) for value in raw_path):
                raise TypeError(f"explicit path {index} must contain three integers")
            normalized.append(tuple(raw_path))
        object.__setattr__(self, "paths", tuple(normalized))

    def select(self, candidates, context: PathSelectionContext):
        del candidates, context
        return (AngularPath(*path) for path in self.paths)


path_policy_registry.register("full_admissible", FullAdmissiblePolicy)
path_policy_registry.register("zipped", ZippedPolicy)
path_policy_registry.register("explicit", ExplicitPathPolicy)
