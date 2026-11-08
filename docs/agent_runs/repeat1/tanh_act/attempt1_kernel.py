import torch
import triton
import triton.language as tl


@triton.jit
def _tanh_bf16_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # Numerically stable tanh: tanh(|x|) = (1 - t) / (1 + t) with t = exp(-2|x|) in (0, 1].
    ax = tl.abs(x)
    t = tl.exp(-2.0 * ax)
    y = (1.0 - t) / (1.0 + t)
    y = tl.where(x < 0.0, -y, y)

    tl.store(out_ptr + offsets, y.to(out_ptr.dtype.element_ty), mask=mask)


TRITON_KERNELS = [_tanh_bf16_kernel]

_BLOCK_SIZE = 1024


def run(*tensors: torch.Tensor) -> torch.Tensor:
    a = tensors[0]
    out = torch.empty(a.shape, dtype=a.dtype, device=a.device)
    x = a.contiguous().view(-1)
    o = out.view(-1)
    n = x.numel()
    grid = (triton.cdiv(n, _BLOCK_SIZE),)
    _tanh_bf16_kernel[grid](x, o, n, BLOCK_SIZE=_BLOCK_SIZE)
    return out
