from __future__ import annotations

import random
from dataclasses import dataclass

from .registry import BenchmarkExample, register
from .code_eval import code_eval_enabled, extract_python_code, run_code_in_subprocess
from ..agents import SubtaskType
from ..tasks import Subtask


@dataclass(frozen=True)
class MBPP:
    name: str = "mbpp"

    def load(self, *, data_dir: str, limit: int, seed: int) -> list[BenchmarkExample]:
        from datasets import load_dataset

        ds = load_dataset("mbpp", split="test")
        rnd = random.Random(seed)
        idxs = list(range(len(ds)))
        rnd.shuffle(idxs)
        idxs = idxs[: max(1, limit)]

        out: list[BenchmarkExample] = []
        for i in idxs:
            ex = ds[int(i)]
            prompt = ex["text"]
            tests = ex.get("test_list") or []
            query = (
                "Write Python code to solve the task.\n"
                "Return ONLY Python code (no markdown).\n\n"
                f"Task:\n{prompt}\n"
            )
            out.append(
                BenchmarkExample(
                    benchmark=self.name,
                    example_id=str(ex.get("task_id", i)),
                    query=query,
                    reference={"prompt": prompt, "tests": tests},
                )
            )
        return out

    def score(self, *, example: BenchmarkExample, final_answer: str | None) -> dict:
        code = extract_python_code(final_answer or "")
        tests = example.reference.get("tests") or []
        tests = [t for t in tests if isinstance(t, str)]

        if code_eval_enabled() and tests and code.strip():
            harness = "\n".join(tests) + "\n"
            r = run_code_in_subprocess(code, harness, timeout_s=3.0)
            return {"metric": "pass@1", "score": 1.0 if r.passed else 0.0, "error": r.error}

        if not code:
            return {"metric": "heuristic_code_present", "score": 0.0}
        return {"metric": "heuristic_code_present", "score": 1.0 if "def " in code else 0.5}

    def make_subtasks(self, *, example: BenchmarkExample) -> list[Subtask]:
        return [
            Subtask(
                id="t1",
                task_type=SubtaskType.CODE,
                description=example.query,
                estimated_tokens=1200,
                difficulty=3.0,
            )
        ]


register(MBPP())

