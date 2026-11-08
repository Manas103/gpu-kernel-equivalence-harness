"""The deterministic control plane. Runs inside the WSL venv as a
subprocess, one call per (task, attempt): import the model's generated
kernel file, run it against the task's fixed-seed inputs, diff against
the fp64 oracle, time it against the eager baseline, and emit one
structured JSON result. This process never calls an LLM and never grades
anything by asking a model whether output "looks right"; every verdict
here is a fixed, re-runnable numeric comparison.

Usage: verify_task.py <task_name> <dtype: fp16|bf16> <kernel_file.py> <out_json>
"""
import importlib.util
import json
import sys
import time
import traceback

import torch

sys.path.insert(0, __file__.rsplit("agent", 1)[0])
from agent import tasks as task_mod  # noqa: E402
from agent import tolerance  # noqa: E402
from agent import metrics  # noqa: E402

SEED = 42
N_TIMING_REPEATS = 30
N_WARMUP = 5


def _time_cuda(fn, *args):
    for _ in range(N_WARMUP):
        fn(*args)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times_ms = []
    for _ in range(N_TIMING_REPEATS):
        start.record()
        fn(*args)
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))
    times_ms.sort()
    return times_ms[len(times_ms) // 2] / 1000.0  # median, seconds


def main():
    task_name, dtype_name, kernel_file, out_json = sys.argv[1:5]
    result = {"task": task_name, "dtype": dtype_name, "status": None}
    try:
        task = task_mod.TASKS_BY_NAME[task_name]
        dtype = task_mod.TORCH_DTYPES[dtype_name]
        inputs = task.inputs(dtype, "cuda", SEED)

        spec = importlib.util.spec_from_file_location("candidate_kernel", kernel_file)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception:
            result["status"] = "compile_error"
            result["error"] = traceback.format_exc()[-4000:]
            _write(out_json, result)
            return

        if not hasattr(mod, "run"):
            result["status"] = "compile_error"
            result["error"] = "generated module has no top-level run() function"
            _write(out_json, result)
            return

        try:
            candidate = mod.run(*inputs)
            torch.cuda.synchronize()
        except Exception:
            result["status"] = "runtime_error"
            result["error"] = traceback.format_exc()[-4000:]
            _write(out_json, result)
            return

        inputs_fp64 = [t.double() for t in inputs]
        oracle_out = task.oracle(inputs_fp64)
        candidate_fp64 = candidate.double()

        if candidate_fp64.shape != oracle_out.shape:
            result["status"] = "fail"
            result["error"] = f"shape mismatch: candidate {tuple(candidate_fp64.shape)} vs oracle {tuple(oracle_out.shape)}"
            _write(out_json, result)
            return

        ok, max_abs_diff, eps = tolerance.within_budget(candidate_fp64, oracle_out, dtype, task.k)
        result["status"] = "pass" if ok else "fail"
        result["max_abs_diff"] = max_abs_diff
        result["tolerance_eps"] = eps

        elapsed_candidate = _time_cuda(mod.run, *inputs)
        elapsed_eager = _time_cuda(task.eager, *inputs)
        result["elapsed_candidate_s"] = elapsed_candidate
        result["elapsed_eager_s"] = elapsed_eager
        result["speedup_over_eager"] = elapsed_eager / elapsed_candidate if elapsed_candidate > 0 else None

        bm = metrics.bytes_moved(list(inputs), candidate)
        flops = None
        if task.category == "matmul":
            M, K = inputs[0].shape
            _, Ndim = inputs[1].shape
            flops = 2 * M * K * Ndim
        result["speed_of_light_proxy"] = metrics.speed_of_light_proxy(elapsed_candidate, bm, flops)

        kernels = getattr(mod, "TRITON_KERNELS", [])
        occ = []
        for kfn in kernels:
            usage = metrics.extract_kernel_resource_usage(kfn)
            if usage is not None:
                n_regs, shared, num_warps = usage
                occ.append(metrics.theoretical_occupancy(n_regs, shared, num_warps))
        result["occupancy_proxy"] = occ if occ else "unavailable (kernel metadata not exposed by this Triton build)"

    except Exception:
        result["status"] = "harness_error"
        result["error"] = traceback.format_exc()[-4000:]

    _write(out_json, result)


def _write(path, result):
    with open(path, "w", newline="\n") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
