"""Quantize/dequantize round-trip and step-size checks, independent of any
GPU kernel: these must hold on the CPU alone, since a bug here would
corrupt both the kernel's input and the oracle's input identically and no
kernel-vs-oracle comparison could ever catch it."""
import torch

from harness.quant import (
    quantize_fp8_groupwise, dequantize_fp8_groupwise,
    quantize_int4_groupwise, dequantize_int4_groupwise,
    fp8_quant_step, int4_quant_step,
    FP8_FORMAT_MAX,
)


def test_fp8_round_trip_within_one_quant_step():
    torch.manual_seed(0)
    x = torch.randn(8, 64) * 3.0
    for fmt in ("fp8_e4m3", "fp8_e5m2"):
        q, scale = quantize_fp8_groupwise(x, group_size=32, fmt=fmt)
        deq = dequantize_fp8_groupwise(q, scale, group_size=32)
        step = fp8_quant_step(scale, fmt)
        # Every dequantized element must be within its own group's worst-case
        # step of the original value (this is a per-group bound, so compare
        # against the group's step broadcast, not a single scalar).
        step_full = step.repeat_interleave(32, dim=-1)
        assert torch.all((deq - x).abs() <= step_full * 1.01), f"{fmt} round trip exceeded its own quant step"


def test_int4_round_trip_within_half_scale():
    torch.manual_seed(1)
    x = torch.randn(8, 64) * 2.0
    q, scale = quantize_int4_groupwise(x, group_size=32)
    deq = dequantize_int4_groupwise(q, scale, group_size=32)
    step = int4_quant_step(scale)
    step_full = step.repeat_interleave(32, dim=-1)
    assert torch.all((deq - x).abs() <= step_full * 1.01)
    assert torch.all(q >= -8) and torch.all(q <= 7)


def test_fp8_e5m2_has_a_looser_step_than_e4m3_at_the_same_scale():
    # e5m2 has 2 mantissa bits against e4m3's 3, so its relative step is
    # coarser at a matched scale.
    scale = torch.tensor([1.0])
    assert fp8_quant_step(scale, "fp8_e5m2") > fp8_quant_step(scale, "fp8_e4m3")


def test_fp8_format_max_constants_match_the_ocp_spec():
    assert FP8_FORMAT_MAX["fp8_e4m3"] == 448.0
    assert FP8_FORMAT_MAX["fp8_e5m2"] == 57344.0


def test_quantize_clips_rather_than_overflows_at_the_format_boundary():
    # A value far beyond the group's own range cannot occur from amax-based
    # scaling (the max element always maps to exactly the format max), but
    # a group with one huge outlier must still clip every other element's
    # *quantized* code to the representable range, never silently wrap.
    x = torch.tensor([[1000.0] + [0.01] * 31])
    q, scale = quantize_int4_groupwise(x, group_size=32)
    assert q[0, 0] == 7  # the outlier maps to the top level by construction
    assert torch.all(q[0, 1:] >= -8) and torch.all(q[0, 1:] <= 7)
