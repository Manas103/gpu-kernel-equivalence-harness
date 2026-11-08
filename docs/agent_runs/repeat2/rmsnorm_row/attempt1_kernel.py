import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_row_kernel(
    x_ptr,
    y_ptr,
    x_s0,
    x_s1,
    y_s0,
    y_s1,
    N,
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(x_ptr + row * x_s0 + cols * x_s1, mask=mask, other=0.0).to(tl.float32)
    mean_sq = tl.sum(x * x, axis=0) / N
    inv_rms = tl.rsqrt(mean_sq + eps)
    y = x * inv_rms
    tl.store(y_ptr + row * y_s0 + cols * y_s1, y.to(tl.float16), mask=mask)


TRITON_KERNELS = [_rmsnorm_row_kernel]


def run(*tensors: torch.Tensor) -> torch.Tensor:
    x = tensors[0]
    M, N = x.shape
    out = torch.empty((M, N), dtype=x.dtype, device=x.device)
    BLOCK_N = triton.next_power_of_2(N)
    num_warps = 8 if BLOCK_N >= 2048 else 4
    _rmsnorm_row_kernel[(M,)](
        x,
        out,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        N,
        1e-5,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
    )
    return out
