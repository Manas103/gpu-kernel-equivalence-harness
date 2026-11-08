import torch
import triton
import triton.language as tl


@triton.jit
def leaky_relu_kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.where(x > 0.0, x, x * 0.01)
    tl.store(out_ptr + offsets, y.to(out_ptr.dtype.element_ty), mask=mask)


TRITON_KERNELS = [leaky_relu_kernel]


def run(*tensors: torch.Tensor) -> torch.Tensor:
    x = tensors[0].contiguous()
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    n_elements = x.numel()
    BLOCK = 1024
    grid = (triton.cdiv(n_elements, BLOCK),)
    leaky_relu_kernel[grid](x, out, n_elements, BLOCK=BLOCK)
    return out
