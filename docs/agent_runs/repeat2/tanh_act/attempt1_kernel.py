import torch
import triton
import triton.language as tl


@triton.jit
def tanh_bf16_kernel(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    # Numerically stable tanh: tanh(|x|) = (1 - e) / (1 + e), e = exp(-2|x|)
    ax = tl.abs(x)
    e = tl.exp(-2.0 * ax)
    t = (1.0 - e) / (1.0 + e)
    y = tl.where(x < 0.0, -t, t)
    tl.store(y_ptr + offs, y.to(tl.bfloat16), mask=mask)


TRITON_KERNELS = [tanh_bf16_kernel]


def run(*tensors: torch.Tensor) -> torch.Tensor:
    a = tensors[0]
    x = a.contiguous().view(-1)
    y = torch.empty_like(x)
    n = x.numel()
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    tanh_bf16_kernel[grid](x, y, n, BLOCK=BLOCK)
    return y.view(a.shape)
