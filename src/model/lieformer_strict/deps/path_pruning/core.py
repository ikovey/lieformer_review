"""Policy-neutral infrastructure for selecting outer angular paths."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, order=True)
class AngularPath:
    """A multiplicity-free outer path ``(input, geometry, output)``."""

    input_degree: int
    geometry_degree: int
    output_degree: int


@dataclass(frozen=True)
class PathSelectionContext:
    """All policy-visible bounds; policies never inspect recoupled branches."""

    feature_lmax: int
    geometry_orders: tuple[int, ...]


@runtime_checkable
class OuterPathPolicy(Protocol):
    """Interface implemented by any outer-path pruning strategy."""

    name: str

    def select(
        self,
        candidates: tuple[AngularPath, ...],
        context: PathSelectionContext,
    ) -> Iterable[AngularPath]: ...


PolicyFactory = Callable[..., OuterPathPolicy]


class PathPolicyRegistry:
    """Small explicit registry used by config names and downstream plugins."""

    def __init__(self) -> None:
        self._factories: dict[str, PolicyFactory] = {}

    def register(self, name: str, factory: PolicyFactory, *, replace: bool = False) -> None:
        key = self._normalize_name(name)
        if key in self._factories and not replace:
            raise ValueError(f"outer-path policy {key!r} is already registered")
        self._factories[key] = factory

    def create(self, name: str, options: Mapping[str, Any] | None = None) -> OuterPathPolicy:
        key = self._normalize_name(name)
        if key not in self._factories:
            available = ", ".join(sorted(self._factories))
            raise ValueError(f"unknown outer-path policy {name!r}; available policies: {available}")
        policy = self._factories[key](**dict(options or {}))
        if not isinstance(policy, OuterPathPolicy):
            raise TypeError(f"factory for {key!r} did not return an OuterPathPolicy")
        return policy

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))

    @staticmethod
    def _normalize_name(name: str) -> str:
        key = str(name).strip().lower()
        if not key:
            raise ValueError("outer-path policy name must not be empty")
        return key


path_policy_registry = PathPolicyRegistry()


def resolve_path_policy(
    policy: str | Mapping[str, Any] | OuterPathPolicy,
) -> OuterPathPolicy:
    """Resolve a config string/mapping or accept a policy object directly.

    Mapping form is intentionally generic, e.g. ``{"name": "zipped",
    "include_zero_frequency": False}``.  New registered strategies therefore
    need no change in the strict tensor-product implementation.
    """

    if isinstance(policy, str):
        return path_policy_registry.create(policy)
    if isinstance(policy, Mapping):
        config = dict(policy)
        try:
            name = config.pop("name")
        except KeyError as error:
            raise ValueError("outer-path policy mapping requires a 'name' field") from error
        return path_policy_registry.create(str(name), config)
    if isinstance(policy, OuterPathPolicy):
        return policy
    raise TypeError("outer_path_policy must be a name, a {'name': ...} mapping, or an OuterPathPolicy")


def build_admissible_candidates(context: PathSelectionContext) -> tuple[AngularPath, ...]:
    """Enumerate P1 once, independently of every pruning strategy."""

    candidates: list[AngularPath] = []
    for input_degree in range(context.feature_lmax + 1):
        for geometry_degree in context.geometry_orders:
            output_min = abs(input_degree - geometry_degree)
            output_max = min(input_degree + geometry_degree, context.feature_lmax)
            for output_degree in range(output_min, output_max + 1):
                candidates.append(AngularPath(input_degree, geometry_degree, output_degree))
    return tuple(candidates)


def build_outer_path_specs(
    feature_lmax: int,
    geometry_orders: Iterable[int],
    policy: str | Mapping[str, Any] | OuterPathPolicy,
) -> tuple[AngularPath, ...]:
    """Build P1, apply a policy, and validate the resulting mathematical subset."""

    orders = tuple(sorted(set(int(order) for order in geometry_orders)))
    if feature_lmax < 0:
        raise ValueError(f"feature_lmax must be non-negative, got {feature_lmax}")
    if not orders or orders[0] < 0:
        raise ValueError(f"geometry_orders must be non-empty and non-negative, got {orders}")
    context = PathSelectionContext(int(feature_lmax), orders)
    candidates = build_admissible_candidates(context)
    candidate_set = set(candidates)
    resolved = resolve_path_policy(policy)
    selected = tuple(resolved.select(candidates, context))

    if len(selected) != len(set(selected)):
        raise ValueError(f"outer-path policy {resolved.name!r} returned duplicate paths")
    outside = tuple(path for path in selected if path not in candidate_set)
    if outside:
        raise ValueError(
            f"outer-path policy {resolved.name!r} returned paths outside full_admissible P1: {outside}"
        )
    # Canonical P1 order makes IDs and checkpoints deterministic, regardless
    # of the order in which a custom policy happens to return its subset.
    selected_set = set(selected)
    return tuple(path for path in candidates if path in selected_set)
