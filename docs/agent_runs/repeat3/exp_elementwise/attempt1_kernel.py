import torch
import triton
import triton.language as tl


@triton.jit
def _exp_bf16_kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.exp(x)
    tl.store(out_ptr + offsets, y.to(tl.bfloat16), mask=mask)


TRITON_KERNELS = [_exp_bf16_kernel]


def run(*tensors: torch.Tensor) -> torch.Tensor:
    a = tensors[0]
    out = torch.empty_like(a)
    n = a.numel()
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    _exp_bf16_kernel[grid](a, out, n, BLOCK=BLOCK)
    return out
