"""Diffs the Triton low-bit (FP8 E4M3/E5M2, INT4) kernel against an fp64
oracle that dequantizes the same quantized operands the kernel receives,
under the quantization-step-derived budget (see harness/lowbit_sweep.py
and harness/tolerance.py for why the oracle dequantizes rather than
reusing the pre-quantization data)."""
import pytest
import torch

from harness.lowbit_sweep import make_lowbit_inputs, run_one_lowbit, LOWBIT_DTYPES

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")


@pytest.mark.parametrize("dtype_kind", LOWBIT_DTYPES)
@pytest.mark.parametrize("shape", [(32, 32, 32), (64, 48, 64)])
def test_lowbit_kernel_matches_oracle(dtype_kind, shape):
    m, n, k = shape
    a, b = make_lowbit_inputs(m, n, k, dtype_kind)
    r = run_one_lowbit(m, n, k, dtype_kind, a, b)
    assert r["passed"], (
        f"{dtype_kind} kernel diverged from the dequantize-then-f64 oracle: "
        f"abs={r['max_abs_diff']} atol={r['atol']}"
    )


@pytest.mark.parametrize("dtype_kind", LOWBIT_DTYPES)
def test_lowbit_kernel_deterministic(dtype_kind):
    """Same premise as test_determinism.py for the base kernels: running
    the identical quantized input twice must give a bit-identical result,
    since the sweep's 'false failures' count depends on this."""
    a, b = make_lowbit_inputs(32, 32, 64, dtype_kind)
    r1 = run_one_lowbit(32, 32, 64, dtype_kind, a, b)
    r2 = run_one_lowbit(32, 32, 64, dtype_kind, a, b)
    assert torch.equal(r1["kernel_out"], r2["kernel_out"])
    assert r1["passed"] == r2["passed"]
