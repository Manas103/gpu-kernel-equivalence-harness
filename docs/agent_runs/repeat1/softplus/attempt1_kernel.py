import torch
import triton
import triton.language as tl


@triton.jit
def _softplus_bf16_kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # Numerically stable softplus: max(x, 0) + log(1 + exp(-|x|))
    abs_x = tl.where(x >= 0, x, -x)
    relu_x = tl.where(x > 0, x, 0.0)
    y = relu_x + tl.log(1.0 + tl.exp(-abs_x))

    tl.store(out_ptr + offsets, y.to(tl.bfloat16), mask=mask)


TRITON_KERNELS = [_softplus_bf16_kernel]


def run(*tensors: torch.Tensor) -> torch.Tensor:
    x = tensors[0].contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK = 1024
    grid = (triton.cdiv(n_elements, BLOCK),)
    _softplus_bf16_kernel[grid](x, out, n_elements, BLOCK=BLOCK)
    return out
