import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_row_kernel(
    x_ptr,
    out_ptr,
    n_cols,
    stride_xm,
    stride_om,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * stride_xm + cols, mask=mask, other=float("-inf"))
    x = x.to(tl.float32)

    row_max = tl.max(x, axis=0)
    num = tl.exp(x - row_max)
    num = tl.where(mask, num, 0.0)
    denom = tl.sum(num, axis=0)
    y = num / denom

    tl.store(out_ptr + row * stride_om + cols, y.to(out_ptr.dtype.element_ty), mask=mask)


TRITON_KERNELS = [_softmax_row_kernel]


def run(*tensors: torch.Tensor) -> torch.Tensor:
    a = tensors[0].contiguous()
    M, N = a.shape
    out = torch.empty((M, N), dtype=a.dtype, device=a.device)

    BLOCK_SIZE = triton.next_power_of_2(N)
    num_warps = 4 if BLOCK_SIZE <= 2048 else 8

    _softmax_row_kernel[(M,)](
        a,
        out,
        N,
        a.stride(0),
        out.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return out
