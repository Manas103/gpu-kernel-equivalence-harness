"""Reduction-order tolerance budget, generalized from harness/tolerance.py
to an arbitrary reduction length K (K=1 for a pointwise task with no
reduction at all, in which case this is just a format-precision budget).

Same derivation as the matmul harness: a length-K reduction's forward
error is bounded by roughly (K-1) * u * magnitude, u the format's unit
roundoff. SAFETY=4.0 is the same measured constant used there (see that
repo's README for the sweep that justified it); it is not re-derived here.
"""
import torch

SAFETY = 4.0

_UNIT_ROUNDOFF = {
    torch.float16: 2 ** -11,
    torch.bfloat16: 2 ** -8,
}


def budget(dtype, k, magnitude=1.0):
    u = _UNIT_ROUNDOFF[dtype]
    k_eff = max(k, 1)
    eps = SAFETY * k_eff * u * magnitude
    return max(eps, 4 * u)  # a pointwise task (k=1) still gets a format-precision floor


def within_budget(actual_fp64, expected_fp64, dtype, k):
    magnitude = expected_fp64.abs().mean().item() + 1.0
    eps = budget(dtype, k, magnitude)
    diff = (actual_fp64 - expected_fp64).abs()
    max_abs_diff = diff.max().item()
    ok = bool((diff <= eps).all().item())
    return ok, max_abs_diff, eps
