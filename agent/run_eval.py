"""Top-level orchestrator, run from Windows (so it can call the real
`claude` CLI directly as a subprocess; see README for why the compile/run
step instead shells out to the WSL venv where Triton and CUDA live).

For each of the 24 tasks, at a fixed-per-task dtype (12 fp16, 12 bf16;
see README, "What this is not", for why dtype is fixed per task rather
than both dtypes per task), the model gets up to MAX_ATTEMPTS tries: the
first with just the task description, any retry with the control plane's
structured feedback on the previous attempt appended to the prompt. This
whole sweep is run REPEATS times so the pass rate can be reported as a
band, not a single number.
"""
import json
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from agent import prompts, llm_client  # noqa: E402
from agent.tasks import TASK_NAMES  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS_DIR = REPO_ROOT / "docs"
RUNS_DIR = DOCS_DIR / "agent_runs"
MAX_ATTEMPTS = 2
REPEATS = 3

WSL_DISTRO = "Ubuntu-22.04"
# ~/proj is a no-space symlink into the Windows-side projects/ dir (this tool's
# own wsl.exe invocation mishandles spaces and $(...) in long paths); the venv
# itself lives on native WSL filesystem (~/build), not /mnt/c, per the
# playbook's I/O-heavy-work guidance.
WSL_REPO_PATH = "~/proj/gpu-kernel-equivalence-harness"
WSL_VENV_PYTHON = "~/build/gkh_venv/bin/python"

DTYPE_FOR_TASK = {name: ("fp16" if i % 2 == 0 else "bf16") for i, name in enumerate(TASK_NAMES)}


def run_verifier(task_name, dtype_name, kernel_path, out_json_path):
    win_kernel = str(kernel_path)
    win_out = str(out_json_path)
    cmd = (
        f'cd {WSL_REPO_PATH} && '
        f'{WSL_VENV_PYTHON} -m agent.verify_task {task_name} {dtype_name} '
        f'"{_to_wsl(win_kernel)}" "{_to_wsl(win_out)}"'
    )
    proc = subprocess.run(
        ["wsl.exe", "-d", WSL_DISTRO, "--", "bash", "-lc", cmd],
        capture_output=True, text=True, timeout=120,
    )
    if out_json_path.exists():
        return json.loads(out_json_path.read_text())
    return {"status": "harness_error", "error": f"no output; stdout={proc.stdout[-1000:]} stderr={proc.stderr[-1000:]}"}


def _to_wsl(win_path):
    p = win_path.replace("\\", "/")
    if p[1:3] == ":/":
        p = "/mnt/" + p[0].lower() + p[2:]
    return p


def summarize_feedback(result):
    keep = {k: v for k, v in result.items() if k not in ("task", "dtype")}
    return json.dumps(keep, indent=2)[:2000]


def run_one_episode(task_name, dtype_name, repeat_idx, episode_dir):
    episode_dir.mkdir(parents=True, exist_ok=True)
    feedback = None
    attempts_log = []
    for attempt in range(1, MAX_ATTEMPTS + 1):
        prompt = prompts.build_prompt(task_name, dtype_name, feedback)
        code, transcript, stderr = llm_client.generate_kernel(prompt)
        kernel_path = episode_dir / f"attempt{attempt}_kernel.py"
        kernel_path.write_text(code, encoding="utf-8", newline="\n")
        (episode_dir / f"attempt{attempt}_transcript.txt").write_text(transcript, encoding="utf-8", newline="\n")

        out_json_path = episode_dir / f"attempt{attempt}_result.json"
        result = run_verifier(task_name, dtype_name, kernel_path, out_json_path)
        attempts_log.append({"attempt": attempt, **result})

        if result.get("status") == "pass":
            break
        feedback = summarize_feedback(result)

    final = attempts_log[-1]
    return {
        "task": task_name,
        "dtype": dtype_name,
        "repeat": repeat_idx,
        "attempts_used": len(attempts_log),
        "final_status": final.get("status"),
        "attempts": attempts_log,
    }


def main():
    task_filter = sys.argv[1].split(",") if len(sys.argv) > 1 else TASK_NAMES
    repeats = int(sys.argv[2]) if len(sys.argv) > 2 else REPEATS
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    all_repeats = []
    for repeat_idx in range(1, repeats + 1):
        episodes = []
        for task_name in task_filter:
            dtype_name = DTYPE_FOR_TASK[task_name]
            episode_dir = RUNS_DIR / f"repeat{repeat_idx}" / task_name
            ep = run_one_episode(task_name, dtype_name, repeat_idx, episode_dir)
            episodes.append(ep)
            print(f"[repeat {repeat_idx}] {task_name} ({dtype_name}): {ep['final_status']} in {ep['attempts_used']} attempt(s)")

        passed = sum(1 for e in episodes if e["final_status"] == "pass")
        speedups = [
            e["attempts"][-1]["speedup_over_eager"]
            for e in episodes
            if e["final_status"] == "pass" and e["attempts"][-1].get("speedup_over_eager")
        ]
        median_speedup = sorted(speedups)[len(speedups) // 2] if speedups else None
        repeat_summary = {
            "repeat": repeat_idx,
            "passed": passed,
            "total": len(episodes),
            "median_speedup_over_eager": median_speedup,
            "episodes": episodes,
        }
        all_repeats.append(repeat_summary)
        (RUNS_DIR / f"repeat{repeat_idx}_summary.json").write_text(
            json.dumps(repeat_summary, indent=2), encoding="utf-8", newline="\n"
        )

    overall = {
        "repeats": [{"repeat": r["repeat"], "passed": r["passed"], "total": r["total"],
                      "median_speedup_over_eager": r["median_speedup_over_eager"]} for r in all_repeats],
        "pass_count_band": [r["passed"] for r in all_repeats],
    }
    (DOCS_DIR / "agent_eval_summary.json").write_text(json.dumps(overall, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
