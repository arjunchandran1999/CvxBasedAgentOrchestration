from __future__ import annotations

import os
import re
import sys
import tempfile
import subprocess
from dataclasses import dataclass


def extract_python_code(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    # Prefer fenced code blocks.
    if "```" in t:
        blocks = re.findall(r"```(?:python)?\s*\n([\s\S]*?)```", t, flags=re.IGNORECASE)
        if blocks:
            return blocks[0].strip()
    return t


@dataclass(frozen=True)
class CodeEvalResult:
    passed: bool
    error: str | None = None
    stderr: str | None = None


def run_code_in_subprocess(code: str, harness: str, *, timeout_s: float = 3.0) -> CodeEvalResult:
    """
    Executes code + harness in a separate Python process with a hard timeout.
    WARNING: This executes untrusted code. Use only on trusted machines.
    """
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "runner.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
            f.write("\n\n")
            f.write(harness)
            f.write("\n")
        try:
            p = subprocess.run(
                [sys.executable, path],
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            if p.returncode == 0:
                return CodeEvalResult(passed=True)
            return CodeEvalResult(passed=False, error=f"exit_code_{p.returncode}", stderr=p.stderr[-4000:] if p.stderr else None)
        except subprocess.TimeoutExpired as e:
            return CodeEvalResult(passed=False, error="timeout", stderr=(e.stderr[-4000:] if isinstance(e.stderr, str) else None))
        except Exception as e:
            return CodeEvalResult(passed=False, error=str(e))


def code_eval_enabled() -> bool:
    return os.getenv("SWARM_CODE_EVAL", "").strip().lower() in {"1", "true", "yes", "y", "on"}

