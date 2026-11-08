import torch
import triton
import triton.language as tl


@triton.jit
def _row_max_kernel(x_ptr, out_ptr, n_cols, stride_row, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    acc = tl.full([BLOCK_N], float("-inf"), dtype=tl.float32)
    for start in range(0, n_cols, BLOCK_N):
        cols = start + offs
        mask = cols < n_cols
        vals = tl.load(x_ptr + row * stride_row + cols, mask=mask, other=float("-inf")).to(tl.float32)
        acc = tl.maximum(acc, vals)
    result = tl.max(acc, axis=0)
    tl.store(out_ptr + row, result.to(out_ptr.dtype.element_ty))


TRITON_KERNELS = [_row_max_kernel]


def run(*tensors: torch.Tensor) -> torch.Tensor:
    x = tensors[0].contiguous()
    n_rows, n_cols = x.shape
    out = torch.empty((n_rows,), device=x.device, dtype=x.dtype)
    BLOCK_N = max(16, min(triton.next_power_of_2(n_cols), 1024))
    _row_max_kernel[(n_rows,)](x, out, n_cols, x.stride(0), BLOCK_N=BLOCK_N)
    return out
