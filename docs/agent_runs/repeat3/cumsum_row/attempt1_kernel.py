import torch
import triton
import triton.language as tl


@triton.jit
def _cumsum_row_kernel(
    x_ptr,
    y_ptr,
    n_cols,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols

    x = tl.load(x_ptr + row * stride_xm + offs * stride_xn, mask=mask, other=0.0)
    x = x.to(tl.float32)

    s = tl.cumsum(x, axis=0)

    tl.store(
        y_ptr + row * stride_ym + offs * stride_yn,
        s.to(y_ptr.dtype.element_ty),
        mask=mask,
    )


TRITON_KERNELS = [_cumsum_row_kernel]


def run(*tensors: torch.Tensor) -> torch.Tensor:
    x = tensors[0]
    n_rows, n_cols = x.shape
    y = torch.empty((n_rows, n_cols), dtype=x.dtype, device=x.device)

    BLOCK = triton.next_power_of_2(n_cols)
    grid = (n_rows,)
    _cumsum_row_kernel[grid](
        x,
        y,
        n_cols,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        BLOCK=BLOCK,
    )
    return y
