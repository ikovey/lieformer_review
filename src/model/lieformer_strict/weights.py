"""One learnable channel map per original edge-wise outer path."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch

from .paths import OuterPath


class OuterPathWeights(torch.nn.Module):
    def __init__(self, paths: Iterable[OuterPath], in_channels: int, out_channels: int):
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.params = torch.nn.ParameterDict(
            {
                str(path.path_id): torch.nn.Parameter(torch.empty(self.in_channels, self.out_channels))
                for path in paths
            }
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.in_channels)
        for weight in self.params.values():
            torch.nn.init.uniform_(weight, -bound, bound)

    def for_path(self, path_id: int) -> torch.Tensor:
        return self.params[str(path_id)]
