"""The '0 false failures over 10 repeats' claim is a claim about the
kernels being bit-deterministic on identical input. This checks that
directly, independent of the budget logic."""
import pytest
import torch

from harness.cuda_kernel import cuda_matmul
from harness.triton_kernel import triton_matmul

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")


@pytest.mark.parametrize("dtype_kind", ["fp16", "bf16", "tf32"])
def test_cuda_kernel_is_bit_deterministic(dtype_kind):
    torch.manual_seed(42)
    cast = {"fp16": torch.float16, "bf16": torch.bfloat16, "tf32": torch.float32}[dtype_kind]
    a = torch.randn(64, 48, device="cuda", dtype=cast)
    b = torch.randn(48, 80, device="cuda", dtype=cast)

    first = cuda_matmul(a, b, dtype_kind)
    for _ in range(9):
        again = cuda_matmul(a, b, dtype_kind)
        assert torch.equal(first, again)


@pytest.mark.parametrize("dtype_kind", ["fp16", "bf16", "tf32"])
def test_triton_kernel_is_bit_deterministic(dtype_kind):
    torch.manual_seed(42)
    cast = {"fp16": torch.float16, "bf16": torch.bfloat16, "tf32": torch.float32}[dtype_kind]
    a = torch.randn(64, 48, device="cuda", dtype=cast)
    b = torch.randn(48, 80, device="cuda", dtype=cast)

    first = triton_matmul(a, b, dtype_kind)
    for _ in range(9):
        again = triton_matmul(a, b, dtype_kind)
        assert torch.equal(first, again)
