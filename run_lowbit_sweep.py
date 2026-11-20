#!/usr/bin/env python3
"""The exact reproducible sweep behind the sub-8-bit extension's claims:
FP8 E4M3, FP8 E5M2 and group-wise INT4 Triton kernels, diffed against an
fp64 oracle under the quantization-step-derived budget. Raw output is
committed verbatim at docs/lowbit_sweep_output.txt."""
import torch

from harness import configs
from harness.lowbit_sweep import (
    run_lowbit_sweep, run_lowbit_adversarial_probe, bisect_minimal_k_lowbit,
    LOWBIT_DTYPES, GROUP_SIZE,
)


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"torch: {torch.__version__}, CUDA: {torch.version.cuda}")
    print(f"group_size: {GROUP_SIZE}")
    print(f"sweep grid: {len(configs.SHAPES)} shapes x {len(LOWBIT_DTYPES)} dtypes x "
          f"{configs.REPEATS} repeats = {len(configs.SHAPES) * len(LOWBIT_DTYPES) * configs.REPEATS} runs")
    print()

    summary = run_lowbit_sweep()

    print(f"total runs: {summary['total_runs']}")
    print(f"elapsed: {summary['elapsed_seconds']:.1f}s")
    print(f"false failures (verdict or output changed on an identical repeat): {summary['false_failures']}")
    print(f"divergent (shape, dtype) configurations: {len(summary['divergences'])}")
    print()

    for (m, n, k, dtype_kind) in summary["divergences"]:
        bisected = bisect_minimal_k_lowbit(m, n, k, dtype_kind)
        print(f"DIVERGENCE dtype={dtype_kind} original_shape=({m},{n},{k}) "
              f"-> minimal_failing_shape=({m},{n},{bisected['minimal_k']}) "
              f"max_abs_diff={bisected['max_abs_diff']:.6g} "
              f"monotonic_in_k={bisected['monotonic_in_k']}")

    print()
    print("adversarial probe (cancellation-structured data, not part of the sweep above):")
    adversarial_found = run_lowbit_adversarial_probe()
    print(f"adversarial divergences found: {len(adversarial_found)}")
    for d in adversarial_found:
        bisected = bisect_minimal_k_lowbit(d["m"], d["n"], d["k"], d["dtype"])
        print(f"ADVERSARIAL DIVERGENCE dtype={d['dtype']} original_shape=({d['m']},{d['n']},{d['k']}) "
              f"max_abs_diff={d['max_abs_diff']:.6g} "
              f"-> minimal_failing_shape=({d['m']},{d['n']},{bisected['minimal_k']}) "
              f"monotonic_in_k={bisected['monotonic_in_k']}")

    print()
    total_divergences = len(summary["divergences"]) + len(adversarial_found)
    print(f"=== claim: 9 divergences the FP16 and BF16 arms never reached ===")
    print(f"measured: {total_divergences} total low-bit divergences "
          f"({len(summary['divergences'])} in the main sweep, {len(adversarial_found)} adversarial)")


if __name__ == "__main__":
    main()
