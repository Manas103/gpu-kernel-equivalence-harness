"""A validation harness that only ever reports pass is as suspicious as
one that only ever reports fail: this deliberately corrupts an output by
an amount well past the derived budget and checks that within_budget
actually says so. Without this test, a bug that made within_budget always
return True would pass every other test in this suite silently."""
import torch

from harness.tolerance import within_budget, tolerance_budget


def test_within_budget_flags_a_large_injected_error():
    k = 64
    dtype_kind = "fp16"
    expected = torch.randn(32, 32) * 5.0
    _, atol = tolerance_budget(k, dtype_kind)
    corrupted = expected + (atol * 1000.0)  # far past budget, not a rounding-sized nudge

    passed, abs_diff, rel_diff, rtol, atol_used = within_budget(corrupted, expected, k, dtype_kind)
    assert not passed
    assert abs_diff > atol_used


def test_within_budget_accepts_a_sub_budget_nudge():
    k = 64
    dtype_kind = "fp16"
    expected = torch.randn(32, 32) * 5.0
    _, atol = tolerance_budget(k, dtype_kind)
    nudged = expected + (atol * 0.1)  # well inside budget

    passed, _, _, _, _ = within_budget(nudged, expected, k, dtype_kind)
    assert passed
