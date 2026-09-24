"""Optional node-local architecture ablations; no dependency on E2Former.

S2 uses e3nn's real spherical-harmonic quadrature convention. Its nonlinear
finite-grid transform is approximately, rather than exactly, SO(3)-equivariant.
"""

from __future__ import annotations

import math

import torch
from e3nn import o3
from torch import nn

from ..deps.atomic.module_mlp import qMLP
from ._shared_primitives import scatter_sum


class HighOrderEmbedding(nn.Module):
    """Initialize l>0 from invariant pair features and relative unit directions."""

    def __init__(self, lmax, channels, num_basis, degree_scale=20.0):
        super().__init__()
        if lmax < 1 or not math.isfinite(degree_scale) or degree_scale <= 0:
            raise ValueError("high-order embedding requires lmax>=1 and positive finite degree_scale")
        self.lmax, self.degree_scale = lmax, float(degree_scale)
        self.source_embedding = nn.Embedding(101, num_basis)
        self.target_embedding = nn.Embedding(101, num_basis)
        nn.init.uniform_(self.source_embedding.weight, -0.001, 0.001)
        nn.init.uniform_(self.target_embedding.weight, -0.001, 0.001)
        hidden = min(num_basis, 128)
        self.radial = nn.ModuleList(
            [qMLP([3 * num_basis, hidden, hidden, channels], norm="ln", activation="silu") for _ in range(lmax)]
        )

    def forward(self, z, edge_index, edge_vectors, radial, decay, num_nodes):
        src, dst = edge_index
        pair = torch.cat([radial, self.source_embedding(z[src]), self.target_embedding(z[dst])], -1)
        # No absolute positions: PBC image shifts are already in edge_vectors.
        parts = []
        for l, projection in enumerate(self.radial, start=1):
            harmonics = o3.spherical_harmonics(l, edge_vectors, normalize=True, normalization="norm")
            message = harmonics.unsqueeze(-1) * (projection(pair) * decay).unsqueeze(1)
            parts.append(scatter_sum(message, dst, num_nodes) / self.degree_scale)
        return torch.cat(parts, dim=1)


class DegreeLinear(nn.Module):
    """Independent channel map per degree, with a scalar-only bias."""

    def __init__(self, lmax, channels):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(channels, channels, bias=l == 0) for l in range(lmax + 1)])
        for layer in self.layers:
            nn.init.uniform_(layer.weight, -(channels**-0.5), channels**-0.5)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, h):
        return torch.cat([layer(h[:, l * l : (l + 1) ** 2]) for l, layer in enumerate(self.layers)], dim=1)


class SeparableS2FFN(nn.Module):
    """Degree linear -> separable sphere SiLU -> degree linear, returning delta.

    All magnetic components are retained. Only the grid for the model's full
    lmax is constructed; unused lower-resolution grids are not instantiated.
    """

    def __init__(self, lmax, channels, resolution=18, zero_init=True):
        super().__init__()
        if not isinstance(resolution, int) or resolution % 2 or resolution < 2 * (lmax + 1):
            raise ValueError("s2_grid_resolution must be even and >= 2*(lmax+1)")
        self.lmax, self.resolution = lmax, resolution
        self.linear1 = DegreeLinear(lmax, channels)
        self.scalar_linear = nn.Linear(channels, channels)
        nn.init.zeros_(self.scalar_linear.bias)
        self.linear2 = DegreeLinear(lmax, channels)
        if zero_init:
            for parameter in self.linear2.parameters():
                nn.init.zeros_(parameter)
        # Construct in double so strict-precision tests are not limited by
        # prematurely rounded quadrature constants. Forward follows h.dtype.
        # e3nn's Legendre code generation uses the default dtype for some
        # constants even when To/FromS2Grid receive an explicit dtype.
        default_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            to_grid = o3.ToS2Grid(lmax, (resolution, resolution), normalization="integral", dtype=torch.float64)
            from_grid = o3.FromS2Grid((resolution, resolution), lmax, normalization="integral", dtype=torch.float64)
        finally:
            torch.set_default_dtype(default_dtype)
        self.register_buffer(
            "to_grid", torch.einsum("mbi,am->bai", to_grid.shb, to_grid.sha).reshape(resolution**2, -1)
        )
        self.register_buffer(
            "from_grid",
            torch.einsum("am,mbi->bai", from_grid.sha, from_grid.shb).reshape(resolution**2, -1).T.contiguous(),
        )

    def forward(self, h):
        x = self.linear1(h)
        # The l=0-only response is constant on the sphere and has no l>0
        # projection. Subtract it before summation to avoid isotropic leakage
        # from FP32 quadrature cancellation, especially near zero high irreps.
        isotropic = x[:, :1] * self.to_grid[0, 0].to(x)
        grid = torch.matmul(self.to_grid[:, 1:].to(x), x[:, 1:]) + isotropic
        activated = torch.nn.functional.silu(grid) - torch.nn.functional.silu(isotropic)
        high = torch.matmul(self.from_grid[1:].to(x), activated)
        scalar = torch.nn.functional.silu(self.scalar_linear(h[:, :1]))
        return self.linear2(torch.cat([scalar, high], dim=1))


class ResidualS2FFN(nn.Module):
    """Preserve a baseline FFN and add a degree-wise controlled S2 correction.

    The bounded form limits the correction RMS for each node and degree to a
    fraction of the corresponding input RMS.  A shared scale over all magnetic
    components and channels preserves rotations within that degree.
    """

    def __init__(
        self,
        baseline: nn.Module,
        lmax: int,
        channels: int,
        resolution: int = 18,
        *,
        include_scalar: bool = False,
        gain: float = 0.1,
        relative_cap: float | None = 0.1,
        eps: float = 1e-12,
    ):
        super().__init__()
        if not math.isfinite(gain) or gain <= 0:
            raise ValueError("s2_residual_gain must be positive and finite")
        if relative_cap is not None and (not math.isfinite(relative_cap) or relative_cap <= 0):
            raise ValueError("s2_residual_cap must be null or positive and finite")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("s2_residual_eps must be positive and finite")
        self.baseline = baseline
        self.correction = SeparableS2FFN(lmax, channels, resolution=resolution, zero_init=True)
        self.lmax = lmax
        self.include_scalar = bool(include_scalar)
        self.gain = float(gain)
        self.relative_cap = None if relative_cap is None else float(relative_cap)
        self.eps = float(eps)
        degrees = lmax + 1
        self.register_buffer("_ratio_sum", torch.zeros(degrees), persistent=False)
        self.register_buffer("_ratio_count", torch.zeros(degrees), persistent=False)
        self.register_buffer("_clipped_count", torch.zeros(degrees), persistent=False)

    def _controlled_correction(self, h: torch.Tensor, correction: torch.Tensor):
        parts = []
        ratios = []
        clipped = []
        for degree in range(self.lmax + 1):
            start, stop = degree * degree, (degree + 1) * (degree + 1)
            source = h[:, start:stop]
            delta = correction[:, start:stop]
            if degree == 0 and not self.include_scalar:
                delta = torch.zeros_like(delta)
            source_work = source.float() if source.dtype in (torch.float16, torch.bfloat16) else source
            delta_work = delta.float() if delta.dtype in (torch.float16, torch.bfloat16) else delta
            # The additive term is inside sqrt so the zero-initialized branch
            # has a finite derivative on its first optimizer step.
            source_power = source_work.square().mean(dim=(1, 2))
            delta_power = delta_work.square().mean(dim=(1, 2))
            source_rms = (source_power + self.eps).sqrt()
            delta_rms = (delta_power + self.eps).sqrt()
            was_clipped = torch.zeros_like(source_rms, dtype=torch.bool)
            if self.relative_cap is not None and (degree > 0 or self.include_scalar):
                limit = self.relative_cap * source_rms
                was_clipped = delta_rms > limit
                scale = torch.where(
                    delta_rms > 0,
                    torch.clamp(limit / delta_rms.clamp_min(self.eps), max=1.0),
                    torch.ones_like(delta_rms),
                )
                delta = delta * scale[:, None, None].to(delta.dtype)
                delta_rms = delta_rms * scale
            applied_power = (
                (delta.float() if delta.dtype in (torch.float16, torch.bfloat16) else delta).square().mean(dim=(1, 2))
            )
            ratio = applied_power.sqrt() / source_rms
            ratio = torch.where(source_power > 0, ratio, torch.zeros_like(ratio))
            parts.append(delta)
            ratios.append(ratio)
            clipped.append(was_clipped)
        return torch.cat(parts, dim=1), ratios, clipped

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        correction = self.gain * self.correction(h)
        correction, ratios, clipped = self._controlled_correction(h, correction)
        if not torch.compiler.is_compiling():
            with torch.no_grad():
                for degree, (ratio, was_clipped) in enumerate(zip(ratios, clipped)):
                    self._ratio_sum[degree] += ratio.detach().sum().to(self._ratio_sum)
                    self._ratio_count[degree] += ratio.numel()
                    self._clipped_count[degree] += was_clipped.detach().sum().to(self._clipped_count)
        return self.baseline(h) + correction

    def diagnostics(self, reset: bool = False) -> dict:
        count = self._ratio_count.clamp_min(1)
        result = {
            "mean_relative_rms_by_degree": (self._ratio_sum / count).detach().cpu().tolist(),
            "clipped_fraction_by_degree": (self._clipped_count / count).detach().cpu().tolist(),
            "samples_by_degree": self._ratio_count.detach().cpu().tolist(),
            "gain": self.gain,
            "relative_cap": self.relative_cap,
            "include_scalar": self.include_scalar,
        }
        if reset:
            self._ratio_sum.zero_()
            self._ratio_count.zero_()
            self._clipped_count.zero_()
        return result


class ScalarResidualFFN(nn.Module):
    """Add only the scalar branch used by :class:`ResidualS2FFN`.

    Construction intentionally consumes the same private RNG sequence as a
    ``SeparableS2FFN``.  Consequently its retained scalar parameters are an
    exact initialization match for the scalar part of the full S2 correction,
    while the unused degree maps and S2 grids are absent at training time.
    """

    def __init__(
        self,
        baseline: nn.Module,
        lmax: int,
        channels: int,
        *,
        gain: float = 0.1,
        relative_cap: float | None = 0.1,
        eps: float = 1e-12,
    ):
        super().__init__()
        if not math.isfinite(gain) or gain <= 0:
            raise ValueError("scalar_residual_gain must be positive and finite")
        if relative_cap is not None and (not math.isfinite(relative_cap) or relative_cap <= 0):
            raise ValueError("scalar_residual_cap must be null or positive and finite")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("scalar_residual_eps must be positive and finite")

        self.baseline = baseline
        # Match SeparableS2FFN's draw order exactly. Only the modules used by
        # its scalar path are retained and registered on this module.
        DegreeLinear(lmax, channels)
        self.scalar_linear = nn.Linear(channels, channels)
        nn.init.zeros_(self.scalar_linear.bias)
        output = DegreeLinear(lmax, channels)
        for parameter in output.parameters():
            nn.init.zeros_(parameter)
        self.scalar_output = output.layers[0]

        self.lmax = int(lmax)
        self.gain = float(gain)
        self.relative_cap = None if relative_cap is None else float(relative_cap)
        self.eps = float(eps)
        degrees = lmax + 1
        self.register_buffer("_ratio_sum", torch.zeros(degrees), persistent=False)
        self.register_buffer("_ratio_count", torch.zeros(degrees), persistent=False)
        self.register_buffer("_clipped_count", torch.zeros(degrees), persistent=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        delta = self.gain * self.scalar_output(torch.nn.functional.silu(self.scalar_linear(h[:, :1])))
        source = h[:, :1]
        source_work = source.float() if source.dtype in (torch.float16, torch.bfloat16) else source
        delta_work = delta.float() if delta.dtype in (torch.float16, torch.bfloat16) else delta
        source_power = source_work.square().mean(dim=(1, 2))
        delta_power = delta_work.square().mean(dim=(1, 2))
        source_rms = (source_power + self.eps).sqrt()
        delta_rms = (delta_power + self.eps).sqrt()
        was_clipped = torch.zeros_like(source_rms, dtype=torch.bool)
        if self.relative_cap is not None:
            limit = self.relative_cap * source_rms
            was_clipped = delta_rms > limit
            scale = torch.clamp(limit / delta_rms.clamp_min(self.eps), max=1.0)
            delta = delta * scale[:, None, None].to(delta.dtype)
        applied = delta.float() if delta.dtype in (torch.float16, torch.bfloat16) else delta
        ratio = applied.square().mean(dim=(1, 2)).sqrt() / source_rms
        ratio = torch.where(source_power > 0, ratio, torch.zeros_like(ratio))
        if not torch.compiler.is_compiling():
            with torch.no_grad():
                self._ratio_sum[0] += ratio.detach().sum().to(self._ratio_sum)
                self._ratio_count[0] += ratio.numel()
                self._clipped_count[0] += was_clipped.detach().sum().to(self._clipped_count)
        correction = torch.zeros_like(h)
        correction[:, :1] = delta
        return self.baseline(h) + correction

    def diagnostics(self, reset: bool = False) -> dict:
        count = self._ratio_count.clamp_min(1)
        result = {
            "mean_relative_rms_by_degree": (self._ratio_sum / count).detach().cpu().tolist(),
            "clipped_fraction_by_degree": (self._clipped_count / count).detach().cpu().tolist(),
            "samples_by_degree": self._ratio_count.detach().cpu().tolist(),
            "gain": self.gain,
            "relative_cap": self.relative_cap,
            "include_scalar": True,
            "scalar_only": True,
        }
        if reset:
            self._ratio_sum.zero_()
            self._ratio_count.zero_()
            self._clipped_count.zero_()
        return result
