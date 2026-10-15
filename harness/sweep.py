"""The sweep: for each of 240 (shape, dtype) configurations, generate one
fixed random input, then run every implementation against it 10 times.

Repeats re-run the identical input, not fresh random draws. That is
deliberate: this harness's "0 false failures over 10 repeats" claim is
about the harness's own determinism, not about surviving bad luck on new
data. A validation harness that gives a different pass/fail verdict on the
same input from one run to the next has not proved anything about the
kernel; it has only proved it cannot be trusted. The sweep checks, for
every one of the 2,400 runs, that output is bit-identical to that
configuration's first run and that the budget verdict never flips.
"""
import hashlib
import time

import torch

from . import configs
from .cuda_kernel import cuda_matmul
from .triton_kernel import triton_matmul
from .reference import cublas_matmul
from .tolerance import within_budget


def _seed_for(m, n, k, dtype_kind):
    key = f"{m}-{n}-{k}-{dtype_kind}".encode()
    return int(hashlib.sha256(key).hexdigest()[:8], 16)


def make_inputs(m, n, k, dtype_kind, device="cuda"):
    torch.manual_seed(_seed_for(m, n, k, dtype_kind))
    if dtype_kind == "tf32":
        a = torch.randn(m, k, device=device, dtype=torch.float32)
        b = torch.randn(k, n, device=device, dtype=torch.float32)
    elif dtype_kind == "fp16":
        a = torch.randn(m, k, device=device, dtype=torch.float16)
        b = torch.randn(k, n, device=device, dtype=torch.float16)
    elif dtype_kind == "bf16":
        a = torch.randn(m, k, device=device, dtype=torch.bfloat16)
        b = torch.randn(k, n, device=device, dtype=torch.bfloat16)
    else:
        raise ValueError(dtype_kind)
    return a, b


def run_one(m, n, k, dtype_kind, a, b):
    """Returns a dict of the comparison result for a single run."""
    ref = cublas_matmul(a, b, dtype_kind)
    cuda_out = cuda_matmul(a, b, dtype_kind)
    triton_out = triton_matmul(a, b, dtype_kind)

    cuda_pass, cuda_abs, cuda_rel, rtol, atol = within_budget(cuda_out, ref, k, dtype_kind)
    triton_pass, triton_abs, triton_rel, _, _ = within_budget(triton_out, ref, k, dtype_kind)

    return {
        "cuda_out": cuda_out, "triton_out": triton_out,
        "cuda_pass": cuda_pass, "cuda_abs": cuda_abs, "cuda_rel": cuda_rel,
        "triton_pass": triton_pass, "triton_abs": triton_abs, "triton_rel": triton_rel,
        "rtol": rtol, "atol": atol,
    }


def run_sweep(shapes=None, dtypes=None, repeats=None):
    shapes = shapes if shapes is not None else configs.SHAPES
    dtypes = dtypes if dtypes is not None else configs.DTYPES
    repeats = repeats if repeats is not None else configs.REPEATS

    results = []
    false_failures = 0
    divergences = []  # (m, n, k, dtype, impl)
    total_runs = 0
    t0 = time.time()

    for (m, n, k) in shapes:
        for dtype_kind in dtypes:
            a, b = make_inputs(m, n, k, dtype_kind)
            first = None
            config_had_divergence = {"cuda": False, "triton": False}
            config_had_pass = {"cuda": False, "triton": False}
            for rep in range(repeats):
                total_runs += 1
                r = run_one(m, n, k, dtype_kind, a, b)
                if first is None:
                    first = r
                else:
                    if not torch.equal(r["cuda_out"], first["cuda_out"]):
                        false_failures += 1
                    if not torch.equal(r["triton_out"], first["triton_out"]):
                        false_failures += 1
                    if r["cuda_pass"] != first["cuda_pass"]:
                        false_failures += 1
                    if r["triton_pass"] != first["triton_pass"]:
                        false_failures += 1

                for impl in ("cuda", "triton"):
                    if r[f"{impl}_pass"]:
                        config_had_pass[impl] = True
                    else:
                        config_had_divergence[impl] = True

            for impl in ("cuda", "triton"):
                if config_had_divergence[impl]:
                    divergences.append((m, n, k, dtype_kind, impl))

            results.append({
                "m": m, "n": n, "k": k, "dtype": dtype_kind,
                "cuda_pass": first["cuda_pass"], "cuda_abs": first["cuda_abs"], "cuda_rel": first["cuda_rel"],
                "triton_pass": first["triton_pass"], "triton_abs": first["triton_abs"], "triton_rel": first["triton_rel"],
                "rtol": first["rtol"], "atol": first["atol"],
            })

    elapsed = time.time() - t0
    return {
        "results": results,
        "divergences": divergences,
        "false_failures": false_failures,
        "total_runs": total_runs,
        "elapsed_seconds": elapsed,
    }
