"""Cancellation-structured inputs, built specifically to make reduction
order matter, unlike the main sweep's i.i.d. random Gaussian data (see
tolerance.py for why that distinction turned out to matter).

Each row of A alternates +1000 and -1000 (plus a small amount of noise so
not every term is bit-identical), and B is all ones, so the exact dot
product is approximately zero: K/2 terms around +1000 and K/2 around
-1000 nearly cancel. A reduction that sums left to right loses relative
precision catastrophically near a near-zero result; a reduction that
restructures the sum (blocked, pairwise, tensor-core-tiled) can land on a
measurably different near-zero value. This is the textbook cancellation
construction for stress-testing summation algorithms, not a contrived
edge case built to make any particular implementation fail.
"""
import torch

ADVERSARIAL_K_VALUES = [128, 512, 4096]
ADVERSARIAL_SHAPE = (64, 64)  # (M, N); K varies


def adversarial_inputs(m: int, n: int, k: int, dtype_kind: str, seed: int = 1):
    torch.manual_seed(seed)
    idx = torch.arange(k, dtype=torch.float32)
    sign = torch.where(idx % 2 == 0, 1.0, -1.0)
    row = sign * 1000.0 + torch.randn(k) * 0.01
    a = row.unsqueeze(0).repeat(m, 1)
    b = torch.ones(k, n, dtype=torch.float32)

    cast = {"fp16": torch.float16, "bf16": torch.bfloat16, "tf32": torch.float32}[dtype_kind]
    return a.to("cuda").to(cast), b.to("cuda").to(cast)
