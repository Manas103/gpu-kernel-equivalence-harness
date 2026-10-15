"""A Triton-compiled matmul kernel, written for this project (not copied
from Triton's own tutorial), used as the third implementation the harness
compares against the hand-written CUDA kernel and the cuBLAS reference."""
import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    USE_TF32: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        a_mask = (rm[:, None] < M) & (rk[None, :] + k0 < K)
        b_mask = (rk[:, None] + k0 < K) & (rn[None, :] < N)
        a_ptrs = a_ptr + rm[:, None] * stride_am + (rk[None, :] + k0) * stride_ak
        b_ptrs = b_ptr + (rk[:, None] + k0) * stride_bk + rn[None, :] * stride_bn
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        if USE_TF32:
            acc += tl.dot(a, b, input_precision="tf32")
        else:
            acc += tl.dot(a, b)

    c_ptrs = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def triton_matmul(a: torch.Tensor, b: torch.Tensor, dtype_kind: str) -> torch.Tensor:
    assert a.is_cuda and b.is_cuda
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    block_m, block_n, block_k = 16, 16, 16
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    _matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        USE_TF32=(dtype_kind == "tf32"),
    )
    return c
