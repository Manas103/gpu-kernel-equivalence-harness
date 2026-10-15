"""Diffs the hand-written CUDA kernel and the Triton kernel against a
deliberately slow, obviously correct oracle: numpy-style float64 matmul on
the CPU, sharing no code with either GPU implementation (see
harness/oracle.py)."""
import pytest
import torch

from harness.cuda_kernel import cuda_matmul
from harness.triton_kernel import triton_matmul
from harness.oracle import oracle_matmul_f64
from harness.tolerance import within_budget

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")


@pytest.mark.parametrize("dtype_kind", ["fp16", "bf16", "tf32"])
@pytest.mark.parametrize("shape", [(32, 32, 16), (64, 48, 96)])
def test_cuda_kernel_matches_oracle(dtype_kind, shape):
    m, n, k = shape
    torch.manual_seed(0)
    cast = {"fp16": torch.float16, "bf16": torch.bfloat16, "tf32": torch.float32}[dtype_kind]
    a = torch.randn(m, k, device="cuda", dtype=cast)
    b = torch.randn(k, n, device="cuda", dtype=cast)

    expected = oracle_matmul_f64(a, b).to("cuda")
    actual = cuda_matmul(a, b, dtype_kind)

    passed, abs_diff, rel_diff, rtol, atol = within_budget(actual, expected, k, dtype_kind)
    assert passed, f"cuda kernel diverged from f64 oracle: abs={abs_diff} rel={rel_diff} budget rtol={rtol} atol={atol}"


@pytest.mark.parametrize("dtype_kind", ["fp16", "bf16", "tf32"])
@pytest.mark.parametrize("shape", [(32, 32, 16), (64, 48, 96)])
def test_triton_kernel_matches_oracle(dtype_kind, shape):
    m, n, k = shape
    torch.manual_seed(0)
    cast = {"fp16": torch.float16, "bf16": torch.bfloat16, "tf32": torch.float32}[dtype_kind]
    a = torch.randn(m, k, device="cuda", dtype=cast)
    b = torch.randn(k, n, device="cuda", dtype=cast)

    expected = oracle_matmul_f64(a, b).to("cuda")
    actual = triton_matmul(a, b, dtype_kind)

    passed, abs_diff, rel_diff, rtol, atol = within_budget(actual, expected, k, dtype_kind)
    assert passed, f"triton kernel diverged from f64 oracle: abs={abs_diff} rel={rel_diff} budget rtol={rtol} atol={atol}"
