"""The sub-8-bit sweep: FP8 E4M3, FP8 E5M2 and group-wise INT4, each diffed
against an fp64 CPU oracle, under the quantization-step-derived budget in
tolerance.py.

Unlike the base sweep (fp16/bf16/tf32, where the operands ARE the reduced
dtype directly), here the pipeline is: generate fp32 data -> quantize to
the target format with a group-size-32 scale (quant.py) -> run the Triton
dequantize-and-matmul kernel on the quantized operands -> compare against
an oracle that dequantizes the SAME quantized operands and matmuls them in
fp64. This isolates "does the kernel's dequantize-and-accumulate math
diverge from an exact computation of the same (already lossy) inputs" from
"how much does quantization itself cost in accuracy", which is a different
question this project does not claim to answer (project 1 in this role's
meta.json does).
"""
import hashlib
import time

import torch

from . import configs
from .quant import (
    quantize_fp8_groupwise, quantize_int4_groupwise,
    fp8_quant_step, int4_quant_step,
)
from .lowbit_kernel import lowbit_matmul
from .tolerance import within_quant_budget

GROUP_SIZE = 32
LOWBIT_DTYPES = ["fp8_e4m3", "fp8_e5m2", "int4"]


def _seed_for(m, n, k, dtype_kind):
    key = f"lowbit-{m}-{n}-{k}-{dtype_kind}".encode()
    return int(hashlib.sha256(key).hexdigest()[:8], 16)


def quantize_operands(a_fp32: torch.Tensor, b_fp32: torch.Tensor, dtype_kind: str, group_size: int = GROUP_SIZE):
    """a_fp32: (M, K), b_fp32: (K, N). B is grouped along its own K axis
    (dim 0), handled by transposing, quantizing like A, and transposing
    back, so both operands use the identical group-wise scheme."""
    if dtype_kind in ("fp8_e4m3", "fp8_e5m2"):
        fmt = dtype_kind
        a_q, scale_a = quantize_fp8_groupwise(a_fp32, group_size, fmt)
        b_q_t, scale_b_t = quantize_fp8_groupwise(b_fp32.t().contiguous(), group_size, fmt)
        step_a = fp8_quant_step(scale_a, fmt)
        step_b = fp8_quant_step(scale_b_t, fmt)
    elif dtype_kind == "int4":
        a_q, scale_a = quantize_int4_groupwise(a_fp32, group_size)
        b_q_t, scale_b_t = quantize_int4_groupwise(b_fp32.t().contiguous(), group_size)
        step_a = int4_quant_step(scale_a)
        step_b = int4_quant_step(scale_b_t)
    else:
        raise ValueError(dtype_kind)

    b_q = b_q_t.t().contiguous()
    scale_b = scale_b_t  # (N, K // group_size); kernel wants (K // group_size, N)
    scale_b = scale_b.t().contiguous()
    return a_q, scale_a, b_q, scale_b, step_a, step_b.t().contiguous()


def lowbit_oracle_f64(a_q, scale_a, b_q, scale_b, group_size: int) -> torch.Tensor:
    """Dequantizes the SAME quantized operands the kernel receives, in
    float64 on the CPU, and matmuls them there. No separate TF32-rounding
    step is applied here, on purpose: this harness's base oracle
    (oracle.py) does not pre-round its TF32 arm's inputs either, because
    TF32 is a tensor-core computation mode applied internally during the
    dot, not a storage format, so there is nothing to pre-round outside
    the kernel. The kernel's own `input_precision="tf32"` is where that
    truncation happens; see lowbit_kernel.py. See module docstring above
    for why the oracle dequantizes rather than reusing the
    pre-quantization fp32 data."""
    a_q_cpu = a_q.detach().to("cpu")
    scale_a_cpu = scale_a.detach().to("cpu", dtype=torch.float64)
    b_q_cpu = b_q.detach().to("cpu")
    scale_b_cpu = scale_b.detach().to("cpu", dtype=torch.float64)

    M, K = a_q_cpu.shape
    _, N = b_q_cpu.shape
    gs = group_size
    a_deq = a_q_cpu.to(torch.float64).reshape(M, K // gs, gs) * scale_a_cpu.unsqueeze(-1)
    a_deq = a_deq.reshape(M, K)
    b_deq = b_q_cpu.to(torch.float64).t().contiguous().reshape(N, K // gs, gs) * scale_b_cpu.t().contiguous().unsqueeze(-1)
    b_deq = b_deq.reshape(N, K).t().contiguous()
    return (a_deq @ b_deq).to(torch.float32)


def make_lowbit_inputs(m, n, k, dtype_kind, device="cuda"):
    torch.manual_seed(_seed_for(m, n, k, dtype_kind))
    a = torch.randn(m, k, device=device, dtype=torch.float32)
    b = torch.randn(k, n, device=device, dtype=torch.float32)
    return a, b


def run_one_lowbit(m, n, k, dtype_kind, a_fp32, b_fp32, group_size: int = GROUP_SIZE):
    a_q, scale_a, b_q, scale_b, step_a, step_b = quantize_operands(a_fp32, b_fp32, dtype_kind, group_size)
    kernel_out = lowbit_matmul(a_q, scale_a, b_q, scale_b, group_size)
    oracle_out = lowbit_oracle_f64(a_q, scale_a, b_q, scale_b, group_size).to(kernel_out.device)

    max_step = torch.maximum(step_a.max(), step_b.max())
    passed, max_abs_diff, atol = within_quant_budget(kernel_out, oracle_out, max_step)
    return {
        "kernel_out": kernel_out, "oracle_out": oracle_out,
        "passed": passed, "max_abs_diff": max_abs_diff, "atol": atol,
    }


def adversarial_lowbit_inputs(m: int, n: int, k: int, seed: int = 1):
    """Same cancellation construction as adversarial.py (alternating
    +1000/-1000 rows against an all-ones B), kept in fp32 so it can be fed
    straight into quantize_operands instead of being cast to a dtype
    directly; the construction is unchanged, only the format pipeline it
    feeds differs."""
    torch.manual_seed(seed)
    idx = torch.arange(k, dtype=torch.float32)
    sign = torch.where(idx % 2 == 0, 1.0, -1.0)
    row = sign * 1000.0 + torch.randn(k) * 0.01
    a = row.unsqueeze(0).repeat(m, 1).to("cuda")
    b = torch.ones(k, n, dtype=torch.float32, device="cuda")
    return a, b


def run_lowbit_adversarial_probe(dtypes=None, group_size: int = GROUP_SIZE):
    """The adversarial.py construction, run through the low-bit pipeline.
    Kept separate from the 2,400-run sweep above, the same discipline the
    base repo applies to its own adversarial probe."""
    dtypes = dtypes if dtypes is not None else LOWBIT_DTYPES
    shapes = [(64, 64, k) for k in (128, 512, 4096)]

    divergences = []
    for (m, n, k) in shapes:
        for dtype_kind in dtypes:
            a, b = adversarial_lowbit_inputs(m, n, k)
            r = run_one_lowbit(m, n, k, dtype_kind, a, b, group_size)
            if not r["passed"]:
                divergences.append({
                    "m": m, "n": n, "k": k, "dtype": dtype_kind,
                    "max_abs_diff": r["max_abs_diff"], "atol": r["atol"],
                })
    return divergences


def bisect_minimal_k_lowbit(m, n, k_start, dtype_kind, group_size: int = GROUP_SIZE, seed: int = 1):
    """Same idea as bisect.py's bisect_minimal_k, re-derived here because
    the low-bit pipeline's inputs are generated differently (quantized,
    not just cast). k_start and the bisected k must stay multiples of
    group_size, since quantize_operands requires that; the search steps in
    units of group_size rather than 1."""
    assert k_start % group_size == 0

    def diverges_at(k):
        a, b = adversarial_lowbit_inputs(m, n, k, seed=seed)
        r = run_one_lowbit(m, n, k, dtype_kind, a, b, group_size)
        return not r["passed"], r["max_abs_diff"]

    lo, hi = 1, k_start // group_size
    best_groups, best_abs = k_start // group_size, None
    while lo <= hi:
        mid = (lo + hi) // 2
        bad, abs_diff = diverges_at(mid * group_size)
        if bad:
            best_groups, best_abs = mid, abs_diff
            hi = mid - 1
        else:
            lo = mid + 1

    best_k = best_groups * group_size
    monotonic_ok = True
    if best_groups > 1:
        bad_below, _ = diverges_at((best_groups - 1) * group_size)
        monotonic_ok = not bad_below

    return {
        "m": m, "n": n, "minimal_k": best_k, "dtype": dtype_kind,
        "max_abs_diff": best_abs, "monotonic_in_k": monotonic_ok,
    }


def run_lowbit_sweep(shapes=None, dtypes=None, repeats=10, group_size: int = GROUP_SIZE):
    shapes = shapes if shapes is not None else configs.SHAPES
    dtypes = dtypes if dtypes is not None else LOWBIT_DTYPES

    results = []
    false_failures = 0
    divergences = []  # (m, n, k, dtype)
    total_runs = 0
    t0 = time.time()

    for (m, n, k) in shapes:
        for dtype_kind in dtypes:
            a, b = make_lowbit_inputs(m, n, k, dtype_kind)
            first = None
            config_had_divergence = False
            for rep in range(repeats):
                total_runs += 1
                r = run_one_lowbit(m, n, k, dtype_kind, a, b, group_size)
                if first is None:
                    first = r
                else:
                    if not torch.equal(r["kernel_out"], first["kernel_out"]):
                        false_failures += 1
                    if r["passed"] != first["passed"]:
                        false_failures += 1
                if not r["passed"]:
                    config_had_divergence = True

            if config_had_divergence:
                divergences.append((m, n, k, dtype_kind))

            results.append({
                "m": m, "n": n, "k": k, "dtype": dtype_kind,
                "passed": first["passed"], "max_abs_diff": first["max_abs_diff"], "atol": first["atol"],
            })

    elapsed = time.time() - t0
    return {
        "results": results,
        "divergences": divergences,
        "false_failures": false_failures,
        "total_runs": total_runs,
        "elapsed_seconds": elapsed,
    }
