from __future__ import annotations

import random
from dataclasses import dataclass

from .registry import BenchmarkExample, register
from .code_eval import code_eval_enabled, extract_python_code, run_code_in_subprocess
from ..agents import SubtaskType
from ..tasks import Subtask


@dataclass(frozen=True)
class HumanEval:
    name: str = "humaneval"

    def load(self, *, data_dir: str, limit: int, seed: int) -> list[BenchmarkExample]:
        from datasets import load_dataset

        ds = load_dataset("openai_humaneval", split="test")
        rnd = random.Random(seed)
        idxs = list(range(len(ds)))
        rnd.shuffle(idxs)
        idxs = idxs[: max(1, limit)]

        out: list[BenchmarkExample] = []
        for i in idxs:
            ex = ds[int(i)]
            prompt = ex["prompt"]
            entry = ex.get("entry_point") or ""
            query = (
                "Write the missing Python code to satisfy the specification.\n"
                "Return ONLY Python code (no markdown).\n\n"
                f"{prompt}\n"
            )
            out.append(
                BenchmarkExample(
                    benchmark=self.name,
                    example_id=str(ex.get("task_id", i)),
                    query=query,
                    reference={
                        "entry_point": entry,
                        "prompt": prompt,
                        "test": ex.get("test"),
                    },
                )
            )
        return out

    def score(self, *, example: BenchmarkExample, final_answer: str | None) -> dict:
        code = extract_python_code(final_answer or "")
        entry = str(example.reference.get("entry_point") or "").strip()
        test = example.reference.get("test")

        if code_eval_enabled() and entry and isinstance(test, str) and test.strip() and code.strip():
            harness = (
                "import sys\n"
                "ns = globals()\n"
                f"candidate = ns.get({entry!r})\n"
                "if candidate is None:\n"
                "    raise RuntimeError('missing_entry_point')\n"
                + "\n"
                + test
                + "\n"
                "check(candidate)\n"
            )
            r = run_code_in_subprocess(code, harness, timeout_s=3.0)
            return {"metric": "pass@1", "score": 1.0 if r.passed else 0.0, "error": r.error}

        # Safe default heuristic: checks that the named function appears.
        if not code:
            return {"metric": "heuristic_contains_def", "score": 0.0}
        if entry and f"def {entry}" in code:
            return {"metric": "heuristic_contains_def", "score": 1.0}
        if "def " in code:
            return {"metric": "heuristic_contains_def", "score": 0.5}
        return {"metric": "heuristic_contains_def", "score": 0.0}

    def make_subtasks(self, *, example: BenchmarkExample) -> list[Subtask]:
        return [
            Subtask(
                id="t1",
                task_type=SubtaskType.CODE,
                description=example.query,
                estimated_tokens=1200,
                difficulty=3.5,
            )
        ]


register(HumanEval())

