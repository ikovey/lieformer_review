import math

import torch
import e3nn

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


class ExpNormalSmearing(torch.nn.Module):
    """
    Exponential Normal Smearing for radial basis functions.

    Uses exponentially spaced means and Gaussian functions for smearing distances.

    Args:
        cutoff (float, optional): Cutoff distance. Defaults to 5.0.
        n_rbf (int, optional): Number of radial basis functions. Defaults to 50.
        trainable (bool, optional): If True, means and betas are learnable parameters. Defaults to False.
    """

    def __init__(self, n_rbf=50, cutoff=5.0, trainable=False):
        super(ExpNormalSmearing, self).__init__()
        if isinstance(cutoff, torch.Tensor):
            cutoff = cutoff.item()
        self.cutoff = cutoff
        self.n_rbf = n_rbf
        self.trainable = trainable

        self.cutoff_fn = CosineCutoff(cutoff)
        self.alpha = 5.0 / cutoff

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", torch.nn.Parameter(means))
            self.register_parameter("betas", torch.nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        start_value = torch.exp(torch.scalar_tensor(-self.cutoff))
        means = torch.linspace(start_value, 1, self.n_rbf)
        betas = torch.tensor([(2 / self.n_rbf * (1 - start_value)) ** -2] * self.n_rbf)
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist):
        """
        Args:
            dist (|E|, 1):  should be exactly 2-dim.

        Returns:
            (|E|, n_basis)
        """
        assert dist.dim() == 2, "dist should be 2-dim"
        _exp = (torch.exp(self.alpha * (-dist)) - self.means) ** 2
        expnorm = torch.exp(-self.betas * _exp)
        return self.cutoff_fn(dist) * expnorm
