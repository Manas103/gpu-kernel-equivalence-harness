import torch
import triton
import triton.language as tl


@triton.jit
def relu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x32 = x.to(tl.float32)
    # ReLU in fp32; NaN inputs pass through unchanged to match torch.relu semantics.
    y32 = tl.where(x32 > 0.0, x32, tl.where(x32 != x32, x32, 0.0))
    tl.store(out_ptr + offsets, y32.to(x.dtype), mask=mask)


TRITON_KERNELS = [relu_kernel]


def run(*tensors: torch.Tensor) -> torch.Tensor:
    a = tensors[0]
    out = torch.empty(a.shape, dtype=a.dtype, device=a.device)
    n_elements = a.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    relu_kernel[grid](a, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out
