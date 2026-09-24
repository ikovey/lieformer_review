"""Cutoff functions."""

import math
import warnings

import torch
from torch import Tensor


class PolynomialCutoff(torch.nn.Module):
    """
    Polynomial cutoff function, as proposed in DimeNet.

    Smoothly reduces values to zero based on a cutoff radius using a polynomial decay.
    Reference: https://arxiv.org/abs/2003.03123

    Args:
        cutoff (float): Cutoff radius.
        p (int, optional): Exponent for the polynomial decay. Defaults to 6.
    """

    def __init__(self, cutoff, p: int = 6):
        super(PolynomialCutoff, self).__init__()
        self.cutoff = cutoff
        self.p = p

    @staticmethod
    def polynomial_cutoff(r: Tensor, rcut: float, p: float = 6.0) -> Tensor:
        """
        Polynomial cutoff, as proposed in DimeNet: https://arxiv.org/abs/2003.03123
        """
        if not p >= 2.0:
            # replace below with logger error
            warnings.warn(f"Exponent p={p} has to be >= 2.\nExiting code.")
            exit()

        rscaled = r / rcut

        out = 1.0
        out = out - (((p + 1.0) * (p + 2.0) / 2.0) * torch.pow(rscaled, p))
        out = out + (p * (p + 2.0) * torch.pow(rscaled, p + 1.0))
        out = out - ((p * (p + 1.0) / 2) * torch.pow(rscaled, p + 2.0))

        return out * (rscaled < 1.0).float()

    def forward(self, r):
        return self.polynomial_cutoff(r=r, rcut=self.cutoff, p=self.p)

    def __repr__(self):
        return f"{self.__class__.__name__}(cutoff={self.cutoff}, p={self.p})"


class CosineCutoff(torch.nn.Module):
    """
    Cosine cutoff function with automatic dtype handling for .float()/.double().

    Smoothly reduces values to zero based on a cutoff radius using a cosine function.

    Args:
        cutoff (float): Cutoff radius.
    """

    def __init__(self, cutoff, dtype=None):
        super(CosineCutoff, self).__init__()

        if isinstance(cutoff, torch.Tensor):
            cutoff = cutoff.item()
        if dtype is None:
            dtype = torch.get_default_dtype()
        self.cutoff = cutoff
        self.dtype = self._resolve_dtype(dtype)

    def forward(self, distances):
        cutoffs = 0.5 * (torch.cos(distances * math.pi / self.cutoff) + 1.0)
        # cutoffs = torch.where(distances < self.cutoff, cutoffs, 0.0)
        cutoffs *= (distances < self.cutoff).to(self.dtype)
        return cutoffs

    def float(self):
        super().float()
        self.dtype = torch.float32
        return self

    def double(self):
        super().double()
        self.dtype = torch.float64
        return self

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        if "dtype" in kwargs:
            self.dtype = kwargs["dtype"]
        elif len(args) > 0:
            if isinstance(args[0], torch.dtype):
                self.dtype = args[0]
            elif hasattr(args[0], "dtype"):  # .to(tensor)
                self.dtype = args[0].dtype
            else:
                raise ValueError(f"Unsupported arg: {type(args[0])} {args[0]}")
        return self

    @staticmethod
    def _resolve_dtype(dtype):
        if isinstance(dtype, str):
            dtype = dtype.lower()
            if dtype == "float" or dtype == "float32":
                return torch.float32
            elif dtype == "double" or dtype == "float64":
                return torch.float64
            elif dtype == "half" or dtype == "float16":
                return torch.float16
            else:
                raise ValueError(f"Unsupported dtype string: {dtype}")
        elif isinstance(dtype, torch.dtype):
            return dtype
        else:
            raise TypeError(f"Expected str or torch.dtype, got {type(dtype)}")
