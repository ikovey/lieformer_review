"""Cost-aware scoring and exhaustive search for small outer-path spaces."""

from __future__ import annotations

import itertools
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .core import AngularPath, build_outer_path_specs
from .probes import OuterPathProbeGates


Triplet = tuple[int, int, int]


def triplet_token(path: Triplet) -> str:
    return f"{path[0]},{path[1]},{path[2]}"


def parse_triplet_token(token: str) -> Triplet:
    values = tuple(int(value) for value in token.split(","))
    if len(values) != 3:
        raise ValueError(f"invalid path token {token!r}")
    return values


class TaylorScoreAccumulator:
    """Accumulate first-order ``abs(gate * grad)`` probe scores by module."""

    schema_version = 1

    def __init__(self) -> None:
        self._sums: dict[str, dict[Triplet, float]] = {}
        self._updates: dict[str, int] = {}

    def update(self, model: torch.nn.Module) -> None:
        found = False
        for module_name, module in model.named_modules():
            if not isinstance(module, OuterPathProbeGates):
                continue
            found = True
            layer = module_name or "<root>"
            layer_scores = self._sums.setdefault(layer, {})
            for key, parameter in module.params.items():
                if parameter.grad is None:
                    raise RuntimeError(f"probe gate {layer}.{key} has no gradient")
                triplet = module.triplets()[key]
                score = float((parameter.detach() * parameter.grad.detach()).abs().cpu())
                layer_scores[triplet] = layer_scores.get(triplet, 0.0) + score
            self._updates[layer] = self._updates.get(layer, 0) + 1
        if not found:
            raise RuntimeError("model contains no OuterPathProbeGates")

    def result(self) -> dict[str, Any]:
        per_layer: dict[str, dict[str, float]] = {}
        normalized_layers: list[dict[Triplet, float]] = []
        for layer in sorted(self._sums):
            updates = self._updates[layer]
            averaged = {path: value / updates for path, value in self._sums[layer].items()}
            total = sum(averaged.values())
            normalized = {
                path: (value / total if total > 0.0 else 0.0) for path, value in averaged.items()
            }
            normalized_layers.append(normalized)
            per_layer[layer] = {triplet_token(path): averaged[path] for path in sorted(averaged)}

        all_paths = sorted(set().union(*(scores.keys() for scores in normalized_layers)))
        aggregate = {
            triplet_token(path): sum(scores.get(path, 0.0) for scores in normalized_layers)
            / len(normalized_layers)
            for path in all_paths
        }
        return {
            "schema_version": self.schema_version,
            "score": "abs(gate*dL_dgate)",
            "layer_normalization": "unit_sum_then_mean",
            "updates": dict(sorted(self._updates.items())),
            "per_layer": per_layer,
            "aggregate": aggregate,
        }


def internal_path_costs(feature_lmax: int, geometry_orders: tuple[int, ...]) -> dict[Triplet, int]:
    """Count nonzero complete 6j branches for every full-admissible path."""

    # Local import avoids making the policy package depend on recoupling at import time.
    from ...paths import OuterPath, recouple_outer_path

    specs = build_outer_path_specs(feature_lmax, geometry_orders, "full_admissible")
    costs: dict[Triplet, int] = {}
    for path_id, spec in enumerate(specs):
        path = OuterPath(path_id, spec.input_degree, spec.geometry_degree, spec.output_degree)
        triplet = (spec.input_degree, spec.geometry_degree, spec.output_degree)
        costs[triplet] = len(recouple_outer_path(path))
    return costs


@dataclass(frozen=True)
class PathCandidate:
    paths: tuple[Triplet, ...]
    internal_cost: int
    importance: float

    def policy_config(self) -> dict[str, Any]:
        return {"name": "explicit", "paths": [list(path) for path in self.paths]}


def _structurally_valid(paths: tuple[Triplet, ...], feature_lmax: int) -> bool:
    outputs = {path[2] for path in paths}
    return outputs == set(range(feature_lmax + 1)) and any(path[1] > 0 for path in paths)


def enumerate_budget_candidates(
    scores: dict[Triplet, float],
    costs: dict[Triplet, int],
    budgets: tuple[int, ...],
    *,
    top_k: int = 3,
    min_budget_fraction: float = 0.8,
) -> dict[int, tuple[PathCandidate, ...]]:
    """Exhaustively rank legal subsets for each internal-branch budget."""

    if set(scores) != set(costs):
        raise ValueError("scores and costs must describe the same canonical path set")
    if top_k <= 0 or not 0.0 <= min_budget_fraction <= 1.0:
        raise ValueError("top_k must be positive and min_budget_fraction must be in [0, 1]")
    paths = tuple(sorted(scores))
    feature_lmax = max(path[0] for path in paths)
    ranked: dict[int, list[PathCandidate]] = {int(budget): [] for budget in budgets}
    for keep in range(1, len(paths) + 1):
        for subset in itertools.combinations(paths, keep):
            if not _structurally_valid(subset, feature_lmax):
                continue
            cost = sum(costs[path] for path in subset)
            importance = sum(scores[path] for path in subset)
            for budget in ranked:
                if budget * min_budget_fraction <= cost <= budget:
                    ranked[budget].append(PathCandidate(subset, cost, importance))
    result: dict[int, tuple[PathCandidate, ...]] = {}
    for budget, candidates in ranked.items():
        candidates.sort(key=lambda item: (-item.importance, -item.internal_cost, item.paths))
        result[budget] = tuple(candidates[:top_k])
    return result


def random_budget_matched_candidates(
    costs: dict[Triplet, int],
    target_cost: int,
    *,
    count: int,
    seed: int,
    min_budget_fraction: float = 0.8,
) -> tuple[PathCandidate, ...]:
    """Draw reproducible legal random controls near a target branch budget."""

    paths = tuple(sorted(costs))
    feature_lmax = max(path[0] for path in paths)
    legal: list[tuple[Triplet, ...]] = []
    for keep in range(1, len(paths) + 1):
        for subset in itertools.combinations(paths, keep):
            cost = sum(costs[path] for path in subset)
            if (
                target_cost * min_budget_fraction <= cost <= target_cost
                and _structurally_valid(subset, feature_lmax)
            ):
                legal.append(subset)
    if count > len(legal):
        raise ValueError(f"requested {count} random candidates, but only {len(legal)} are legal")
    rng = random.Random(seed)
    selected = rng.sample(legal, count)
    return tuple(
        PathCandidate(paths=subset, internal_cost=sum(costs[path] for path in subset), importance=0.0)
        for subset in selected
    )


def write_search_manifest(
    path: str | Path,
    *,
    metadata: dict[str, Any],
    candidates: dict[int, tuple[PathCandidate, ...]],
) -> None:
    """Write a deterministic, auditable JSON artifact for static retraining."""

    payload = {
        "schema_version": 1,
        "metadata": metadata,
        "candidates": {
            str(budget): [asdict(candidate) | {"policy": candidate.policy_config()} for candidate in values]
            for budget, values in sorted(candidates.items())
        },
    }
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
