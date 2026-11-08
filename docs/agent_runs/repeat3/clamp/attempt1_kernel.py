import torch
import triton
import triton.language as tl


@triton.jit
def _clamp_fp16_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    min_val,
    max_val,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x32 = x.to(tl.float32)
    y32 = tl.minimum(tl.maximum(x32, min_val), max_val)
    tl.store(out_ptr + offsets, y32.to(tl.float16), mask=mask)


TRITON_KERNELS = [_clamp_fp16_kernel]


def run(*tensors: torch.Tensor) -> torch.Tensor:
    a = tensors[0]
    n_elements = a.numel()
    out = torch.empty_like(a)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _clamp_fp16_kernel[grid](
        a,
        out,
        n_elements,
        -1.0,
        1.0,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out
