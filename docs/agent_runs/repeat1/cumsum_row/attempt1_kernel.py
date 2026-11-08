import torch
import triton
import triton.language as tl


@triton.jit
def _cumsum_row_kernel(
    x_ptr, out_ptr,
    N,
    stride_xm, stride_xn,
    stride_om, stride_on,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(x_ptr + row * stride_xm + cols * stride_xn, mask=mask, other=0.0).to(tl.float32)
    y = tl.cumsum(x, axis=0)
    tl.store(out_ptr + row * stride_om + cols * stride_on, y, mask=mask)


TRITON_KERNELS = [_cumsum_row_kernel]


def run(*tensors: torch.Tensor) -> torch.Tensor:
    x = tensors[0]
    M, N = x.shape
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    BLOCK_N = triton.next_power_of_2(N)
    _cumsum_row_kernel[(M,)](
        x, out, N,
        x.stride(0), x.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_N=BLOCK_N,
    )
    return out
