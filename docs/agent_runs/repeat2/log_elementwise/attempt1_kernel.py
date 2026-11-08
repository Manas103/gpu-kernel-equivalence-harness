import torch
import triton
import triton.language as tl


@triton.jit
def log_elementwise_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    y = tl.log(x)
    tl.store(out_ptr + offsets, y.to(tl.float16), mask=mask)


TRITON_KERNELS = [log_elementwise_kernel]


def run(*tensors: torch.Tensor) -> torch.Tensor:
    x = tensors[0].contiguous()
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    log_elementwise_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out
