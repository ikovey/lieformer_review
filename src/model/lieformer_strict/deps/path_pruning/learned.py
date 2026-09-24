"""Differentiable, budget-aware gates for V3-B path discovery."""

from __future__ import annotations

import itertools
from collections.abc import Iterable

import torch

from .probes import path_key, path_triplet


class LearnedOuterPathGates(torch.nn.Module):
    """One continuous gate per complete outer path.

    This module is a search-time instrument: multiplying by a near-zero gate
    does not skip work.  Exported paths must be rebuilt with an explicit path
    policy to obtain an actual speedup.
    """

    def __init__(
        self,
        paths: Iterable,
        costs: dict[tuple[int, int, int], int],
        target_cost: float | None = None,
        temperature: float = 1.0,
        init_logit: float = 5.0,
    ) -> None:
        super().__init__()
        paths = tuple(paths)
        self.temperature = float(temperature)
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        self.params = torch.nn.ParameterDict(
            {
                path_key(path): torch.nn.Parameter(torch.tensor(float(init_logit)))
                for path in paths
            }
        )
        self._triplets = {path_key(path): path_triplet(path) for path in paths}
        missing = [key for key, triplet in self._triplets.items() if triplet not in costs]
        if missing:
            raise ValueError(f"missing internal costs for paths: {missing}")
        self.register_buffer(
            "cost_vector",
            torch.tensor(
                [float(costs[triplet]) for triplet in self._triplets.values()],
                dtype=torch.float32,
            ),
        )
        self.target_cost = None if target_cost is None else float(target_cost)

    def probabilities(self) -> torch.Tensor:
        return torch.sigmoid(torch.stack(tuple(self.params.values())) / self.temperature)

    def for_path(self, path) -> torch.Tensor:
        return torch.sigmoid(self.params[path_key(path)] / self.temperature)

    def expected_cost(self) -> torch.Tensor:
        probabilities = self.probabilities()
        return (probabilities * self.cost_vector.to(probabilities)).sum()

    def budget_penalty(self) -> torch.Tensor:
        """Squared relative excess above the requested internal-path budget."""
        expected = self.expected_cost()
        if self.target_cost is None:
            return expected.new_zeros(())
        scale = max(float(self.cost_vector.sum()), 1.0)
        return (torch.relu(expected - self.target_cost) / scale).square()

    def triplets(self) -> dict[str, tuple[int, int, int]]:
        return dict(self._triplets)

    def export(self, budget: int) -> dict:
        """Solve the small discrete budget problem and return an explicit policy."""
        triplets = tuple(self._triplets.values())
        probabilities = self.probabilities().detach().cpu().tolist()
        costs = self.cost_vector.detach().cpu().tolist()
        required_outputs = set(range(max(path[2] for path in triplets) + 1))
        best = None
        for keep in range(1, len(triplets) + 1):
            for indices in itertools.combinations(range(len(triplets)), keep):
                cost = int(sum(costs[index] for index in indices))
                selected = tuple(triplets[index] for index in indices)
                if cost > budget or {path[2] for path in selected} != required_outputs:
                    continue
                score = sum(probabilities[index] for index in indices)
                candidate = (score, cost, selected)
                if best is None or candidate > best:
                    best = candidate
        if best is None:
            raise ValueError(f"no structurally valid path subset fits budget {budget}")
        score, cost, selected = best
        ranked = sorted(
            (
                {"path": list(path), "probability": float(probability), "cost": int(path_cost)}
                for path, probability, path_cost in zip(triplets, probabilities, costs)
            ),
            key=lambda item: (-item["probability"], item["path"]),
        )
        return {
            "policy": {"name": "explicit", "paths": [list(path) for path in selected]},
            "internal_cost": cost,
            "probability_sum": score,
            "gates": ranked,
        }
