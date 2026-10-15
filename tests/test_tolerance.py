from harness.tolerance import tolerance_budget, UNIT_ROUNDOFF


def test_tf32_and_fp16_share_the_same_unit_roundoff():
    # Both formats have 10 explicit mantissa bits.
    assert UNIT_ROUNDOFF["fp16"] == UNIT_ROUNDOFF["tf32"]


def test_bf16_has_a_looser_unit_roundoff_than_fp16():
    assert UNIT_ROUNDOFF["bf16"] > UNIT_ROUNDOFF["fp16"]


def test_budget_grows_linearly_with_k():
    rtol_32, _ = tolerance_budget(32, "fp16")
    rtol_64, _ = tolerance_budget(64, "fp16")
    # (64-1)/(32-1) ~= 2.03, so the budget should be very close to double.
    assert 1.9 < rtol_64 / rtol_32 < 2.2


def test_budget_never_negative_or_zero_for_k_equals_one():
    rtol, atol = tolerance_budget(1, "fp16")
    assert rtol > 0 and atol > 0
