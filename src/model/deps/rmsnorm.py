import torch


def get_l_to_all_m_expand_index(lmax: int) -> torch.Tensor:
    expand_index = torch.empty((lmax + 1) ** 2, dtype=torch.long)
    start = 0
    for l in range(lmax + 1):
        width = 2 * l + 1
        expand_index[start : start + width] = l
        start += width
    return expand_index


class EquivariantRMSNorm(torch.nn.Module):
    def __init__(self, lmax, dn, eps=1.0e-8, bias_l0=True):
        super().__init__()
        self.lmax = lmax
        self.dn = dn
        self.eps = eps
        self.scale = torch.nn.Parameter(torch.ones(lmax + 1, dn))
        self.bias_l0 = bias_l0
        if bias_l0:
            self.l0_bias = torch.nn.Parameter(torch.zeros(dn))
        else:
            self.register_parameter("l0_bias", None)

    def forward(self, a_hi):
        outputs = []
        start = 0
        for l in range(self.lmax + 1):
            width = 2 * l + 1
            chunk = a_hi[:, start : start + width, :]
            rms = torch.rsqrt(chunk.pow(2).mean(dim=1, keepdim=True) + self.eps)
            chunk = chunk * rms * self.scale[l].view(1, 1, self.dn)
            if l == 0 and self.l0_bias is not None:
                chunk = chunk + self.l0_bias.view(1, 1, self.dn)
            outputs.append(chunk)
            start += width
        return torch.cat(outputs, dim=1)


class EquivariantRMSNorm_expand(torch.nn.Module):
    def __init__(
        self,
        lmax,
        dn,
        eps=1.0e-8,
        bias_l0=True,
    ):
        super().__init__()
        self.lmax = lmax
        self.dn = dn
        self.eps = eps
        self.scale = torch.nn.Parameter(torch.ones(lmax + 1, dn))
        self.bias_l0 = bias_l0

        self.register_buffer("expand_index", get_l_to_all_m_expand_index(lmax))
        self.register_buffer(
            "inv_multiplicity",
            torch.tensor([1.0 / (2 * l + 1) for l in range(lmax + 1)], dtype=torch.float32),
        )

        if bias_l0:
            self.l0_bias = torch.nn.Parameter(torch.zeros(dn))
        else:
            self.register_parameter("l0_bias", None)

    def forward(self, a_hi):
        sq = a_hi.pow(2)
        reduce_index = self.expand_index.view(1, -1, 1).expand(a_hi.shape[0], -1, self.dn)
        sq_by_l = a_hi.new_zeros((a_hi.shape[0], self.lmax + 1, self.dn))
        sq_by_l.scatter_add_(1, reduce_index, sq)

        rms = torch.rsqrt(sq_by_l * self.inv_multiplicity.view(1, -1, 1).to(a_hi.dtype) + self.eps)
        scaled_rms = rms * self.scale.view(1, self.lmax + 1, self.dn)
        expanded_scale = scaled_rms.index_select(1, self.expand_index)

        out = a_hi * expanded_scale
        if self.l0_bias is not None:
            out[:, 0:1, :] = out[:, 0:1, :] + self.l0_bias.view(1, 1, self.dn)

        return out


class EquivariantRMSNorm_global(torch.nn.Module):
    """Global spherical-harmonic RMSNorm with optional ablation switches.

    Input is treated as ``[..., (lmax + 1)^2, dn]`` and flattened over leading
    batch dimensions internally.

    Ablation switches:
    - ``balance_degrees``:
      If True, aggregate degree contributions with a fixed per-degree weighting
      scheme and then average across degrees. If False, use a plain mean over
      all spherical-harmonic components.
    - ``div_width``:
      Only used when ``balance_degrees=True``. If True, each degree-l block is
      divided by its width ``2l+1`` before the cross-degree averaging, so every
      degree contributes equally regardless of how many m components it has. If
      False, every (l, m) component inside the balanced path has unit weight.
    - ``affine``:
      If True, apply a learnable per-degree scale after the shared global RMS
      factor is computed. If False, use only the shared RMS factor.
    - ``bias_l0``:
      If True, add a learnable bias only to the l=0 block after normalization.
    """

    def __init__(
        self,
        lmax,
        dn,
        eps=1.0e-8,
        balance_degrees=True,
        affine=True,
        bias_l0=True,
        div_width=True,
    ):
        super().__init__()
        self.lmax = lmax
        self.dn = dn
        self.eps = eps
        self.balance_degrees = balance_degrees
        self.affine = affine
        self.bias_l0 = bias_l0
        self.div_width = div_width

        self.register_buffer("expand_index", get_l_to_all_m_expand_index(lmax))

        if affine:
            self.scale = torch.nn.Parameter(torch.ones(lmax + 1, dn))
        else:
            self.register_parameter("scale", None)
        if bias_l0:
            self.l0_bias = torch.nn.Parameter(torch.zeros(dn))
        else:
            self.register_parameter("l0_bias", None)

        if balance_degrees:
            degree_weight = torch.empty((lmax + 1) ** 2, dtype=torch.float32)
            start = 0
            for l in range(lmax + 1):
                width = 2 * l + 1
                degree_weight[start : start + width] = 1.0 / width if div_width else 1.0
                start += width
            degree_weight /= lmax + 1
            self.register_buffer("degree_weight", degree_weight)
        else:
            self.register_buffer("degree_weight", None)

    def forward(self, a_hi):
        out_shape = a_hi.shape[:-2]
        feature = a_hi.reshape(out_shape.numel(), (self.lmax + 1) ** 2, self.dn)

        sq = feature.pow(2)
        if self.balance_degrees:
            norm_sq = torch.einsum("nmc,m->nc", sq, self.degree_weight.to(feature.dtype))
        else:
            norm_sq = sq.mean(dim=1)

        rms = torch.rsqrt(norm_sq.mean(dim=1, keepdim=True).unsqueeze(1) + self.eps)
        if self.affine:
            scaled_rms = rms * self.scale.index_select(0, self.expand_index).unsqueeze(0)
        else:
            scaled_rms = rms

        out = feature * scaled_rms

        if self.l0_bias is not None:
            out[:, 0:1, :] = out[:, 0:1, :] + self.l0_bias.view(1, 1, self.dn)

        return out.reshape(out_shape + ((self.lmax + 1) ** 2, self.dn))
