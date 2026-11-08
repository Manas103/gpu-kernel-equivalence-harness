import torch
import triton
import triton.language as tl


@triton.jit
def _row_max_kernel(x_ptr, out_ptr, N, stride_xm, stride_xn, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + row * stride_xm + offs * stride_xn, mask=mask, other=float("-inf"))
    m = tl.max(x.to(tl.float32), axis=0)
    tl.store(out_ptr + row, m.to(out_ptr.dtype.element_ty))


TRITON_KERNELS = [_row_max_kernel]


def run(*tensors: torch.Tensor) -> torch.Tensor:
    x = tensors[0]
    M, N = x.shape
    out = torch.empty((M,), dtype=x.dtype, device=x.device)
    BLOCK_N = triton.next_power_of_2(N)
    _row_max_kernel[(M,)](x, out, N, x.stride(0), x.stride(1), BLOCK_N=BLOCK_N, num_warps=4)
    return out
