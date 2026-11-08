import torch
import triton
import triton.language as tl


@triton.jit
def _elementwise_sub_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    c = a - b
    tl.store(c_ptr + offsets, c.to(c_ptr.dtype.element_ty), mask=mask)


TRITON_KERNELS = [_elementwise_sub_kernel]


def run(*tensors: torch.Tensor) -> torch.Tensor:
    a, b = tensors[0], tensors[1]
    assert a.is_cuda and b.is_cuda
    assert a.shape == b.shape
    a = a.contiguous()
    b = b.contiguous()
    c = torch.empty_like(a)
    n_elements = a.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _elementwise_sub_kernel[grid](a, b, c, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return c
