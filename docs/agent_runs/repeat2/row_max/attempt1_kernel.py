import torch
import triton
import triton.language as tl


@triton.jit
def _row_max_kernel(
    x_ptr,
    out_ptr,
    N,
    stride_xm,
    stride_xn,
    stride_om,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    row_base = x_ptr + row * stride_xm
    acc = tl.full((BLOCK_N,), float("-inf"), dtype=tl.float32)
    for start in range(0, N, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        mask = cols < N
        vals = tl.load(row_base + cols * stride_xn, mask=mask, other=float("-inf")).to(tl.float32)
        acc = tl.maximum(acc, vals)
    row_max = tl.max(acc, axis=0)
    tl.store(out_ptr + row * stride_om, row_max.to(out_ptr.dtype.element_ty))


TRITON_KERNELS = [_row_max_kernel]


def run(*tensors: torch.Tensor) -> torch.Tensor:
    x = tensors[0]
    M, N = x.shape
    out = torch.empty((M,), device=x.device, dtype=x.dtype)
    BLOCK_N = min(triton.next_power_of_2(N), 4096)
    grid = (M,)
    _row_max_kernel[grid](
        x, out, N,
        x.stride(0), x.stride(1),
        out.stride(0),
        BLOCK_N=BLOCK_N,
    )
    return out
