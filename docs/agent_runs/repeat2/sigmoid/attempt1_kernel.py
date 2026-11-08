import torch
import triton
import triton.language as tl


@triton.jit
def _sigmoid_fp16_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + offsets, y.to(tl.float16), mask=mask)


TRITON_KERNELS = [_sigmoid_fp16_kernel]


def run(*tensors: torch.Tensor) -> torch.Tensor:
    a = tensors[0]
    n_elements = a.numel()
    out = torch.empty(a.shape, dtype=a.dtype, device=a.device)

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _sigmoid_fp16_kernel[grid](
        a.reshape(-1),
        out.reshape(-1),
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out
