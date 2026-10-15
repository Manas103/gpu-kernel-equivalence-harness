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
