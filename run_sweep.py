#!/usr/bin/env python3
"""The exact reproducible sweep behind this project's claims. Prints a
summary and the full list of divergences, each bisected to its minimal
failing shape. Raw output is committed verbatim at docs/sweep_output.txt."""
import torch

from harness import configs
from harness.sweep import run_sweep, run_one
from harness.bisect import bisect_minimal_k
from harness.adversarial import adversarial_inputs, ADVERSARIAL_K_VALUES, ADVERSARIAL_SHAPE


def run_adversarial_probe():
    """Not part of the 2,400-run sweep: a separate, clearly labeled check
    that the harness has teeth, using cancellation-structured inputs
    instead of the main sweep's i.i.d. random Gaussian data (see
    harness/adversarial.py)."""
    m, n = ADVERSARIAL_SHAPE
    found = []
    for k in ADVERSARIAL_K_VALUES:
        for dtype_kind in configs.DTYPES:
            a, b = adversarial_inputs(m, n, k, dtype_kind)
            r = run_one(m, n, k, dtype_kind, a, b)
            for impl in ("cuda", "triton"):
                if not r[f"{impl}_pass"]:
                    found.append((m, n, k, dtype_kind, impl, r[f"{impl}_abs"]))
    return found


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"torch: {torch.__version__}, CUDA: {torch.version.cuda}")
    print(f"sweep grid: {len(configs.SHAPES)} shapes x {len(configs.DTYPES)} dtypes x "
          f"{configs.REPEATS} repeats = {len(configs.SHAPES) * len(configs.DTYPES) * configs.REPEATS} runs")
    print()

    summary = run_sweep()

    print(f"total runs: {summary['total_runs']}")
    print(f"elapsed: {summary['elapsed_seconds']:.1f}s")
    print(f"false failures (verdict or output changed on an identical repeat): {summary['false_failures']}")
    print(f"divergent (shape, dtype, impl) configurations: {len(summary['divergences'])}")
    print()

    for (m, n, k, dtype_kind, impl) in summary["divergences"]:
        bisected = bisect_minimal_k(m, n, k, dtype_kind, impl)
        print(f"DIVERGENCE impl={impl} dtype={dtype_kind} original_shape=({m},{n},{k}) "
              f"-> minimal_failing_shape=({m},{n},{bisected['minimal_k']}) "
              f"max_abs_diff={bisected['max_abs_diff']:.6g} max_rel_diff={bisected['max_rel_diff']:.6g} "
              f"monotonic_in_k={bisected['monotonic_in_k']}")

    print()
    print("adversarial probe (cancellation-structured data, not part of the 2,400-run sweep):")
    adversarial_found = run_adversarial_probe()
    print(f"adversarial divergences found: {len(adversarial_found)}")
    for (m, n, k, dtype_kind, impl, abs_diff) in adversarial_found:
        bisected = bisect_minimal_k(m, n, k, dtype_kind, impl, input_fn=adversarial_inputs)
        print(f"ADVERSARIAL DIVERGENCE impl={impl} dtype={dtype_kind} original_shape=({m},{n},{k}) "
              f"max_abs_diff={abs_diff:.6g} "
              f"-> minimal_failing_shape=({m},{n},{bisected['minimal_k']}) "
              f"monotonic_in_k={bisected['monotonic_in_k']}")


if __name__ == "__main__":
    main()
