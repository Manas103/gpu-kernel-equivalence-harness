import torch
import triton
import triton.language as tl


@triton.jit
def sigmoid_fp16_kernel(
    a_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    a32 = a.to(tl.float32)
    y32 = 1.0 / (1.0 + tl.exp(-a32))
    y = y32.to(tl.float16)
    tl.store(out_ptr + offsets, y, mask=mask)


TRITON_KERNELS = [sigmoid_fp16_kernel]


def run(*tensors: torch.Tensor) -> torch.Tensor:
    a = tensors[0]
    out = torch.empty_like(a)
    n_elements = a.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    sigmoid_fp16_kernel[grid](a, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out
