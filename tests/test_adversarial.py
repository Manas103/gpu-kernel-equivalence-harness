"""Codifies the measured finding from the adversarial probe (see
harness/adversarial.py and the README's Findings section) as a regression
test: cancellation-structured data at large magnitude reveals a real
TF32 divergence past budget that ordinary random Gaussian data never
does, for both the hand-written CUDA kernel and the Triton kernel, while
fp16 and bf16 are unaffected because this specific construction cancels
exactly even at reduced precision."""
import pytest
import torch

from harness.adversarial import adversarial_inputs
from harness.sweep import run_one

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")


def test_tf32_diverges_on_cancellation_data():
    a, b = adversarial_inputs(64, 64, 128, "tf32")
    r = run_one(64, 64, 128, "tf32", a, b)
    assert not r["cuda_pass"]
    assert not r["triton_pass"]


@pytest.mark.parametrize("dtype_kind", ["fp16", "bf16"])
def test_fp16_and_bf16_do_not_diverge_on_the_same_construction(dtype_kind):
    a, b = adversarial_inputs(64, 64, 128, dtype_kind)
    r = run_one(64, 64, 128, dtype_kind, a, b)
    assert r["cuda_pass"]
    assert r["triton_pass"]
