"""Harness self-check for the agent control plane (agent/verify_task.py),
in the same spirit as test_harness_catches_injected_divergence.py: this
never calls an LLM. It feeds the deterministic verifier two hand-written
kernels per task family (one correct, one deliberately wrong) and checks
that the verdict is right both ways, so a bug that made every verdict
"pass" would be caught here rather than silently passing every agent run."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from agent import metrics, tolerance  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

GOOD_VECTOR_ADD = '''
import torch
import triton
import triton.language as tl

@triton.jit
def _add_kernel(a_ptr, b_ptr, c_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    tl.store(c_ptr + offs, a + b, mask=mask)

TRITON_KERNELS = [_add_kernel]

def run(a, b):
    n = a.numel()
    c = torch.empty_like(a)
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    _add_kernel[grid](a, b, c, n, BLOCK=BLOCK)
    return c
'''

BROKEN_VECTOR_ADD = GOOD_VECTOR_ADD.replace("a + b", "a - b")

GOOD_ROW_SUM = '''
import torch
import triton
import triton.language as tl

@triton.jit
def _row_sum_kernel(a_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    a = tl.load(a_ptr + row * n_cols + offs, mask=mask, other=0.0).to(tl.float32)
    s = tl.sum(a, axis=0)
    tl.store(out_ptr + row, s.to(out_ptr.dtype.element_ty))

TRITON_KERNELS = [_row_sum_kernel]

def run(a):
    n_rows, n_cols = a.shape
    out = torch.empty((n_rows,), device=a.device, dtype=a.dtype)
    BLOCK = triton.next_power_of_2(n_cols)
    _row_sum_kernel[(n_rows,)](a, out, n_cols, BLOCK=BLOCK)
    return out
'''

BROKEN_ROW_SUM = GOOD_ROW_SUM.replace("tl.sum(a, axis=0)", "tl.sum(a, axis=0) + 1000.0")


def _run_verifier(tmp_path, task_name, dtype_name, source):
    kernel_file = tmp_path / "kernel.py"
    kernel_file.write_text(source)
    out_json = tmp_path / "result.json"
    subprocess.run(
        [sys.executable, "-m", "agent.verify_task", task_name, dtype_name, str(kernel_file), str(out_json)],
        cwd=str(REPO), check=True, timeout=60,
    )
    return json.loads(out_json.read_text())


def test_verifier_passes_correct_vector_add(tmp_path):
    result = _run_verifier(tmp_path, "vector_add", "fp16", GOOD_VECTOR_ADD)
    assert result["status"] == "pass", result
    assert result["speedup_over_eager"] is not None


def test_verifier_fails_wrong_vector_add(tmp_path):
    result = _run_verifier(tmp_path, "vector_add", "fp16", BROKEN_VECTOR_ADD)
    assert result["status"] == "fail", result
    assert result["max_abs_diff"] > result["tolerance_eps"]


def test_verifier_passes_correct_row_sum(tmp_path):
    result = _run_verifier(tmp_path, "row_sum", "bf16", GOOD_ROW_SUM)
    assert result["status"] == "pass", result


def test_verifier_catches_wrong_row_sum(tmp_path):
    result = _run_verifier(tmp_path, "row_sum", "bf16", BROKEN_ROW_SUM)
    assert result["status"] == "fail", result
    assert result["max_abs_diff"] > result["tolerance_eps"]


def test_verifier_reports_compile_error_on_syntax_error(tmp_path):
    result = _run_verifier(tmp_path, "vector_add", "fp16", "def run(a, b):\n    return a +")
    assert result["status"] == "compile_error"


def test_verifier_reports_runtime_error_on_missing_run(tmp_path):
    result = _run_verifier(tmp_path, "vector_add", "fp16", "x = 1\n")
    assert result["status"] == "compile_error"
    assert "run" in result["error"]


def test_theoretical_occupancy_known_values():
    # 32 regs/thread, 256 threads/block (8 warps), no shared memory:
    # blocks_by_regs = 65536 // (32*256) = 8; blocks_by_threads = 1536//256 = 6
    # -> limited to 6 blocks, 48 warps, which is the full 48-warp SM.
    occ = metrics.theoretical_occupancy(n_regs=32, shared_bytes=0, num_warps=8)
    assert occ["theoretical_occupancy_pct"] == 100.0


def test_theoretical_occupancy_register_bound():
    # 255 regs/thread, 128 threads/block (4 warps): blocks_by_regs = 65536 // (255*128) = 2
    occ = metrics.theoretical_occupancy(n_regs=255, shared_bytes=0, num_warps=4)
    assert occ["blocks_limited_by_registers"] == 2
    assert occ["theoretical_occupancy_pct"] < 50.0


def test_tolerance_budget_scales_with_k():
    eps_k1 = tolerance.budget(torch.float16, k=1)
    eps_k1024 = tolerance.budget(torch.float16, k=1024)
    assert eps_k1024 > eps_k1
