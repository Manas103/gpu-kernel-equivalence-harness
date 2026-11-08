"""Thin wrapper around a real `claude -p` subprocess call. This is the
only thing in the whole loop that calls a model; verify_task.py (the
control plane) never does. Model choice is Haiku 5.5, disclosed in the
README, chosen for cost since this loop makes on the order of a hundred
calls across 24 tasks x 2 dtypes x 3 repeats with retries."""
import re
import subprocess

MODEL = "claude-haiku-5-5"
TIMEOUT_S = 180


def generate_kernel(prompt):
    proc = subprocess.run(
        ["claude", "-p", prompt, "--model", MODEL],
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
    )
    transcript = proc.stdout
    code = _extract_code_block(transcript)
    return code, transcript, proc.stderr


def _extract_code_block(text):
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1)
    return text
