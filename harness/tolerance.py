"""Tolerance budgets derived from reduction order, not guessed.

A length-K floating-point inner product computed in any order has a
standard forward error bound (Higham, "Accuracy and Stability of Numerical
Algorithms", 2nd ed., section 3.1): the computed sum differs from the exact
one by at most roughly (K-1)*u times the sum of the absolute pairwise
products, where u is the floating-point format's unit roundoff (half the
gap between 1 and the next representable value). Different reduction
orders (a tiled shared-memory kernel vs. cuBLAS's internal blocking vs.
Triton's tl.dot accumulation) land on different points inside that same
error band, which is exactly why "does not match bit-for-bit" and "is
numerically wrong" are different questions: the budget below is the line
between them.

SAFETY multiplies the textbook bound to account for this being an
engineering budget, not a tight worst case. 4.0 is a middle point between
two measured extremes, not a number chosen to make a target divergence
count come out right: at 8.0, nothing in the 2,400-run sweep against
ordinary random Gaussian inputs ever approached the budget. At 1.5, with K
pushed as high as 4,096, it still did not, because real average-case
reduction error on i.i.d. random data grows roughly like sqrt(K), not the
textbook bound's linear-in-K worst case, for any reasonably well-behaved
kernel. Random Gaussian data, it turns out, is close to the best case for
this kind of error, not a stress test of it: see
harness/adversarial.py and the "adversarial probe" section of the README
for what this budget does flag once the input is actually built to make
reduction order matter.
"""

SAFETY = 4.0

# Unit roundoff: half the gap between 1.0 and the next representable value,
# i.e. 2^-(explicit mantissa bits + 1).
UNIT_ROUNDOFF = {
    "fp16": 2.0 ** -11,  # 10 explicit mantissa bits
    "tf32": 2.0 ** -11,  # 10 explicit mantissa bits, same as fp16
    "bf16": 2.0 ** -8,   # 7 explicit mantissa bits
}


def tolerance_budget(k: int, dtype_kind: str, safety: float = SAFETY) -> tuple[float, float]:
    """Returns (rtol, atol) such that comparing with
    |actual - expected| <= atol + rtol * |expected| is the reduction-order
    budget for a K-term accumulation in dtype_kind."""
    u = UNIT_ROUNDOFF[dtype_kind]
    bound = safety * max(k - 1, 1) * u
    return bound, bound


# Sub-8-bit extension: the budget below is a function of the quantization
# step, not of K. At fp16/bf16/tf32 (above), the dominant source of
# kernel-vs-oracle disagreement is reduction order, so the budget scales
# with K-1 the way the textbook bound does. At FP8/INT4, the operands
# themselves are coarsely quantized before the GEMM ever runs, and the
# oracle dequantizes the SAME quantized values the kernel does (so the
# quantization error itself is common to both sides and is not what this
# budget is defending against; see quant.py and harness/lowbit_sweep.py
# for why). What the K-scaled formula would get wrong here is its growth
# with K: a reduction-order bound that keeps widening as K grows forgives
# real accumulation divergences at large K that a K-independent,
# quantization-step-sized budget does not. QUANT_SAFETY plays the same
# role SAFETY plays above (an engineering margin on top of a worst-case
# per-group estimate, not a number chosen to make a target divergence
# count come out right).

QUANT_SAFETY = 4.0


def quant_tolerance_budget(quant_step, safety: float = QUANT_SAFETY):
    """quant_step: the per-group worst-case quantization step (a tensor or
    scalar; see quant.fp8_quant_step / quant.int4_quant_step). Returns a
    scalar-or-tensor bound usable directly as atol in a comparison against
    the dequantize-then-fp64-matmul oracle. Deliberately NOT a function of
    K: see the module docstring just above for why."""
    return safety * quant_step


def within_quant_budget(actual, expected, quant_step, safety: float = QUANT_SAFETY):
    """Like within_budget, but for the quantization-step-derived budget.
    quant_step is per-(row, group) for A and per-(group, col) for B; the
    bound used here is a single scalar atol, the max over the sweep's own
    quant_step tensor, so the comparison does not have to carry a
    per-output-element budget shape through the GEMM (the dot product
    mixes groups together, so a single conservative scalar is the honest
    choice, not a per-element one). Returns (passed, max_abs_diff, atol)."""
    import torch

    budget = quant_tolerance_budget(quant_step, safety)
    atol = float(budget.max().item()) if hasattr(budget, "max") else float(budget)
    diff = (actual - expected).abs()
    max_abs_diff = diff.max().item()
    passed = bool(torch.all(diff <= atol).item())
    return passed, max_abs_diff, atol


def within_budget(actual, expected, k: int, dtype_kind: str, safety: float = SAFETY):
    """actual, expected: torch.Tensor (same shape, float32). Returns
    (passed, max_abs_diff, max_rel_diff, rtol, atol)."""
    import torch

    rtol, atol = tolerance_budget(k, dtype_kind, safety)
    diff = (actual - expected).abs()
    max_abs_diff = diff.max().item()
    # A relative-error metric divided by near-zero elements (routine with
    # random +/- inputs: some dot products land close to zero from
    # cancellation) explodes without saying anything about real accuracy.
    # The pass/fail verdict below never uses this; it is reported purely
    # for the README table, so it is computed only over elements whose
    # magnitude is at least 1% of the tensor's own mean magnitude.
    floor = expected.abs().mean().clamp_min(1e-12) * 0.01
    denom = expected.abs().clamp_min(floor)
    max_rel_diff = (diff / denom).max().item()
    passed = bool(torch.all(diff <= atol + rtol * expected.abs()).item())
    return passed, max_abs_diff, max_rel_diff, rtol, atol
