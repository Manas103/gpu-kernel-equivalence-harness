import torch
import triton
import triton.language as tl


@triton.jit
def _layernorm_row_kernel(
    x_ptr,
    y_ptr,
    n_cols,
    stride_x,
    stride_y,
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < n_cols

    x_row = x_ptr + row * stride_x
    y_row = y_ptr + row * stride_y

    x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / n_cols
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)

    y = (x - mean) * rstd
    tl.store(y_row + cols, y.to(tl.bfloat16), mask=mask)


TRITON_KERNELS = [_layernorm_row_kernel]


def run(*tensors: torch.Tensor) -> torch.Tensor:
    x = tensors[0]
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)

    BLOCK_N = triton.next_power_of_2(n_cols)
    grid = (n_rows,)
    _layernorm_row_kernel[grid](
        x,
        out,
        n_cols,
        x.stride(0),
        out.stride(0),
        1e-5,
        BLOCK_N=BLOCK_N,
    )
    return out
