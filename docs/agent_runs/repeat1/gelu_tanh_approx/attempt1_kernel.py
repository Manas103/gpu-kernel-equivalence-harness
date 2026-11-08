import torch
import triton
import triton.language as tl


@triton.jit
def _gelu_tanh_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    inner = 0.7978845608 * (a + 0.044715 * a * a * a)
    # tanh(z) = 1 - 2 / (exp(2z) + 1); saturates cleanly to +/-1 for large |z|
    e = tl.exp(2.0 * inner)
    t = 1.0 - 2.0 / (e + 1.0)
    y = 0.5 * a * (1.0 + t)
    tl.store(out_ptr + offsets, y.to(tl.float16), mask=mask)


TRITON_KERNELS = [_gelu_tanh_kernel]


def run(*tensors: torch.Tensor) -> torch.Tensor:
    x = tensors[0].contiguous()
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    _gelu_tanh_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)
    return out
