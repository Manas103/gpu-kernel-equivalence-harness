import torch
import triton
import triton.language as tl


@triton.jit
def _cumsum_row_kernel(
    x_ptr,
    y_ptr,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    N,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    x_row = x_ptr + row * stride_xm
    y_row = y_ptr + row * stride_ym

    # Block-shaped fp32 carry holding the running sum of all previous chunks.
    carry = tl.zeros([BLOCK], dtype=tl.float32)

    for start in range(0, N, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(x_row + offs * stride_xn, mask=mask, other=0.0).to(tl.float32)
        # Inclusive scan within the chunk, plus the carry from earlier chunks.
        c = tl.cumsum(x, axis=0) + carry
        total = tl.sum(x, axis=0)
        carry = carry + total
        tl.store(
            y_row + offs * stride_yn,
            c.to(y_ptr.dtype.element_ty),
            mask=mask,
        )


TRITON_KERNELS = [_cumsum_row_kernel]


def run(*tensors: torch.Tensor) -> torch.Tensor:
    x = tensors[0].contiguous()
    M, N = x.shape
    y = torch.empty((M, N), dtype=x.dtype, device=x.device)
    grid = (M,)
    _cumsum_row_kernel[grid](
        x,
        y,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        N,
        BLOCK=1024,
    )
    return y
