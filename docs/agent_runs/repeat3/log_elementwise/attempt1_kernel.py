import torch
import triton
import triton.language as tl


@triton.jit
def _log_elementwise_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    y = tl.log(x)
    tl.store(out_ptr + offsets, y.to(out_ptr.dtype.element_ty), mask=mask)


TRITON_KERNELS = [_log_elementwise_kernel]


def run(*tensors: torch.Tensor) -> torch.Tensor:
    a = tensors[0]
    out = torch.empty_like(a)
    n = a.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    _log_elementwise_kernel[grid](a, out, n, BLOCK_SIZE=BLOCK_SIZE)
    return out
