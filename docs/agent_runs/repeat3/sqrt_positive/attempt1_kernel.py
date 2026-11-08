import torch
import triton
import triton.language as tl


@triton.jit
def sqrt_positive_kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    y = tl.sqrt(x)
    tl.store(out_ptr + offsets, y.to(out_ptr.dtype.element_ty), mask=mask)


TRITON_KERNELS = [sqrt_positive_kernel]


def run(*tensors: torch.Tensor) -> torch.Tensor:
    x = tensors[0].contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    sqrt_positive_kernel[grid](x, out, n, BLOCK=BLOCK)
    return out
