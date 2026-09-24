"""Weighted sparse aggregation without materializing an E x moment-width tensor.

The custom backward uses sparse MM for feature gradients and a tiled edge dot
product for attention gradients. In particular it avoids sparse.mm's native
gradient with respect to sparse values, which can form a dense N x N product.
Higher derivatives use the ordinary differentiable gather/scatter expressions.
"""

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


if triton is not None:
    @triton.jit
    def _edge_dot(X, GRAD, SRC, DST, PARTIAL, WIDTH: tl.constexpr,
                  PARTS: tl.constexpr, BLOCK: tl.constexpr):
        edge = tl.program_id(0)
        part = tl.program_id(1)
        c = part * BLOCK + tl.arange(0, BLOCK)
        src = tl.load(SRC + edge)
        dst = tl.load(DST + edge)
        x = tl.load(X + src * WIDTH + c, c < WIDTH, other=0)
        g = tl.load(GRAD + dst * WIDTH + c, c < WIDTH, other=0)
        dot = tl.sum(x * g, axis=0)
        tl.store(PARTIAL + edge * PARTS + part, dot)


class _WeightedAggregate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, alpha, edge_index, num_targets):
        src, dst = edge_index
        matrix = torch.sparse_coo_tensor(
            torch.stack([dst, src]), alpha,
            (num_targets, features.shape[0]), device=features.device,
        )
        ctx.save_for_backward(features, alpha, edge_index)
        return torch.sparse.mm(matrix, features)

    @staticmethod
    def backward(ctx, grad):
        features, alpha, edges = ctx.saved_tensors
        src, dst = edges
        grad_features = grad_alpha = None
        # Ordinary tensor operations retain the graph when create_graph=True.
        higher_order = torch.is_grad_enabled()
        if ctx.needs_input_grad[0]:
            if higher_order:
                grad_features = torch.zeros_like(features).index_add(
                    0, src, grad[dst] * alpha[:, None]
                )
            else:
                transpose = torch.sparse_coo_tensor(
                    torch.stack([src, dst]), alpha,
                    (features.shape[0], grad.shape[0]), device=features.device,
                )
                grad_features = torch.sparse.mm(transpose, grad)
        if ctx.needs_input_grad[1]:
            if (not higher_order and triton is not None and features.is_cuda
                    and features.dtype == torch.float32 and src.numel()):
                width = features.shape[1]
                parts = triton.cdiv(width, 1024)
                partial = features.new_empty((src.numel(), parts))
                _edge_dot[(src.numel(), parts)](
                    features.contiguous(), grad.contiguous(), src.contiguous(), dst.contiguous(),
                    partial, width, parts, 1024,
                )
                grad_alpha = partial.sum(dim=1)
            else:
                # Bound peak memory on CPU and when Triton is unavailable.
                grad_alpha = torch.cat([
                    (features[s] * grad[d]).sum(dim=1)
                    for s, d in zip(src.split(512), dst.split(512))
                ]) if src.numel() else alpha * 0
        return grad_features, grad_alpha, None, None


def weighted_aggregate(features, alpha, edge_index, num_targets):
    """sum_j alpha_ij features_j; includes duplicate edges and empty targets."""
    return _WeightedAggregate.apply(features, alpha, edge_index, num_targets)
