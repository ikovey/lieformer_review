import triton
import triton.language as tl


@triton.jit
def silu_activation_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    SiLU activation function kernel

    Computes the element-wise function: {silu}(x) = x \cdot \mathrm{sigmoid}(x) = \frac{x}{1 + e^{-x}}
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    output = x * (1 / (1 + tl.exp(-x)))
    tl.store(output_ptr + offsets, output, mask=mask)
