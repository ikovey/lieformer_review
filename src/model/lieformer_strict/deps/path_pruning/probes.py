"""Differentiable probes for scoring complete outer angular paths."""

from __future__ import annotations

from collections.abc import Iterable

import torch


def path_triplet(path) -> tuple[int, int, int]:
    """Return the stable mathematical identity of an outer path."""

    return (int(path.input_degree), int(path.geometry_degree), int(path.output_degree))


def path_key(path) -> str:
    input_degree, geometry_degree, output_degree = path_triplet(path)
    return f"a{input_degree}_n{geometry_degree}_l{output_degree}"


class OuterPathProbeGates(torch.nn.Module):
    """One scalar gate per complete outer path, initialized to identity.

    These gates are instrumentation for importance estimation.  They do not
    prune work and therefore must never be used to claim a speedup.
    """

    def __init__(self, paths: Iterable) -> None:
        super().__init__()
        paths = tuple(paths)
        self.params = torch.nn.ParameterDict(
            {path_key(path): torch.nn.Parameter(torch.ones(())) for path in paths}
        )
        self._triplets = {path_key(path): path_triplet(path) for path in paths}

    def for_path(self, path) -> torch.Tensor:
        return self.params[path_key(path)]

    def triplets(self) -> dict[str, tuple[int, int, int]]:
        return dict(self._triplets)
