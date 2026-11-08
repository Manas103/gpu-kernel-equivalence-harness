import torch
import triton
import triton.language as tl


@triton.jit
def matmul_small_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_idx = k0 + offs_k
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (k_idx[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(k_idx[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask)


TRITON_KERNELS = [matmul_small_kernel]

_BLOCK_M = 64
_BLOCK_N = 64
_BLOCK_K = 32


def run(*tensors: torch.Tensor) -> torch.Tensor:
    a, b = tensors[0], tensors[1]
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, "inner dimensions must match"

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    grid = (triton.cdiv(M, _BLOCK_M), triton.cdiv(N, _BLOCK_N))
    matmul_small_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        BLOCK_K=_BLOCK_K,
    )
    return c
