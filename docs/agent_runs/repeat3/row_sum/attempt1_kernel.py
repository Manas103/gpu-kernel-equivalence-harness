import torch
import triton
import triton.language as tl


@triton.jit
def _row_sum_kernel(
    x_ptr,
    out_ptr,
    M,
    N,
    stride_xm,
    stride_xn,
    stride_om,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for start in range(0, N, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        mask = cols < N
        vals = tl.load(x_ptr + row * stride_xm + cols * stride_xn, mask=mask, other=0.0)
        acc += vals.to(tl.float32)
    total = tl.sum(acc, axis=0)
    tl.store(out_ptr + row * stride_om, total.to(out_ptr.dtype.element_ty))


TRITON_KERNELS = [_row_sum_kernel]


def run(*tensors: torch.Tensor) -> torch.Tensor:
    x = tensors[0]
    M, N = x.shape
    out = torch.empty((M,), device=x.device, dtype=x.dtype)
    BLOCK_N = min(triton.next_power_of_2(N), 1024)
    _row_sum_kernel[(M,)](
        x,
        out,
        M,
        N,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        BLOCK_N=BLOCK_N,
    )
    return out
