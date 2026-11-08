import torch
import triton
import triton.language as tl


@triton.jit
def _l2_normalize_row_kernel(
    x_ptr, out_ptr,
    stride_xm, stride_xn,
    stride_om, stride_on,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(x_ptr + row * stride_xm + cols * stride_xn, mask=mask, other=0.0).to(tl.float32)
    sumsq = tl.sum(x * x, axis=0)
    inv_norm = 1.0 / tl.sqrt(sumsq)
    y = x * inv_norm
    tl.store(out_ptr + row * stride_om + cols * stride_on,
             y.to(out_ptr.dtype.element_ty), mask=mask)


TRITON_KERNELS = [_l2_normalize_row_kernel]


def run(*tensors: torch.Tensor) -> torch.Tensor:
    a = tensors[0]
    M, N = a.shape
    out = torch.empty_like(a)
    BLOCK_N = triton.next_power_of_2(N)
    _l2_normalize_row_kernel[(M,)](
        a, out,
        a.stride(0), a.stride(1),
        out.stride(0), out.stride(1),
        N=N, BLOCK_N=BLOCK_N,
    )
    return out
