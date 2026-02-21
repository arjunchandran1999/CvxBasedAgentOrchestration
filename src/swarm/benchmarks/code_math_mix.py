from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from ..agents import SubtaskType
from ..tasks import Subtask
from .code_eval import code_eval_enabled, extract_python_code, run_code_in_subprocess
from .registry import BenchmarkExample, register


def _extract_int(text: str) -> int | None:
    m = re.findall(r"-?\d+", text or "")
    return int(m[-1]) if m else None


def _norm_yesno(text: str) -> str | None:
    t = (text or "").strip().lower()
    if t.startswith("yes") or t == "true":
        return "yes"
    if t.startswith("no") or t == "false":
        return "no"
    return None


@dataclass(frozen=True)
class SingleTask:
    task_id: str
    task_type: SubtaskType
    prompt: str
    expected: Any
    estimated_tokens: int = 300
    difficulty: float = 2.5


class CodeMathMix:
    name = "code_math_mix"

    def __init__(self) -> None:
        self.tasks: list[SingleTask] = [
            SingleTask(
                task_id="t_add",
                task_type=SubtaskType.CODE,
                prompt="Write Python: def add(a,b): return a+b. Return ONLY code.",
                expected={"entry_point": "add", "tests": ["assert add(2,3)==5", "assert add(-1,1)==0"]},
                estimated_tokens=280,
                difficulty=1.5,
            ),
            SingleTask(
                task_id="t_pal",
                task_type=SubtaskType.CODE,
                prompt="Write Python: def is_palindrome(s): return True if s reads same reversed. Return ONLY code.",
                expected={"entry_point": "is_palindrome", "tests": ["assert is_palindrome('racecar')", "assert not is_palindrome('abc')"]},
                estimated_tokens=420,
                difficulty=2.5,
            ),
            SingleTask(
                task_id="t_math1",
                task_type=SubtaskType.MATH,
                prompt="John has 3 apples and buys 4 more. How many apples now? Output only the number.",
                expected=7,
                estimated_tokens=120,
                difficulty=1.0,
            ),
            SingleTask(
                task_id="t_math2",
                task_type=SubtaskType.MATH,
                prompt="Compute: 17*6. Output only the number.",
                expected=102,
                estimated_tokens=120,
                difficulty=1.5,
            ),
            SingleTask(
                task_id="t_reason1",
                task_type=SubtaskType.REASONING,
                prompt="Answer YES or NO only: Is a whale a mammal?",
                expected="yes",
                estimated_tokens=100,
                difficulty=1.5,
            ),
            SingleTask(
                task_id="t_reason2",
                task_type=SubtaskType.REASONING,
                prompt="Answer YES or NO only: Is 2 a prime number?",
                expected="yes",
                estimated_tokens=100,
                difficulty=1.0,
            ),
            SingleTask(
                task_id="t_extract1",
                task_type=SubtaskType.EXTRACTION,
                prompt="Extract JSON from: 'Order: id=42, item=widget'. Return ONLY JSON {\"id\":42,\"item\":\"widget\"}.",
                expected={"id": 42, "item": "widget"},
                estimated_tokens=160,
                difficulty=1.5,
            ),
            SingleTask(
                task_id="t_sum1",
                task_type=SubtaskType.SUMMARIZE,
                prompt="Summarize in one sentence: 'The cat sat on the mat and purred.'",
                expected={"keywords": ["cat", "mat"]},
                estimated_tokens=140,
                difficulty=1.5,
            ),
        ]

    def load(self, *, data_dir: str, limit: int, seed: int) -> list[BenchmarkExample]:
        xs = self.tasks[: max(1, min(limit, len(self.tasks)))]
        out = []
        for t in xs:
            out.append(
                BenchmarkExample(
                    benchmark=self.name,
                    example_id=t.task_id,
                    query=t.prompt,
                    reference={
                        "task_id": t.task_id,
                        "task_type": t.task_type.value,
                        "expected": t.expected,
                        "estimated_tokens": t.estimated_tokens,
                        "difficulty": t.difficulty,
                    },
                )
            )
        return out

    def make_subtasks(self, *, example: BenchmarkExample) -> list[Subtask] | None:
        ref = example.reference
        return [
            Subtask(
                id=str(ref["task_id"]),
                task_type=SubtaskType(str(ref["task_type"])),
                description=str(example.query),
                estimated_tokens=int(ref.get("estimated_tokens", 300)),
                difficulty=float(ref.get("difficulty", 2.5)),
            )
        ]

    def score(self, *, example: BenchmarkExample, final_answer: str | None) -> dict:
        # Not used; score via artifact for direct access to subtask output.
        return {"metric": "accuracy", "score": 0.0}

    def score_artifact(self, *, example: BenchmarkExample, artifact: dict) -> dict:
        ref = example.reference
        expected = ref.get("expected")
        # single-subtask job: output is in results[0]
        results = artifact.get("results") or []
        out = ""
        if results and isinstance(results[0], dict):
            out = str(results[0].get("output") or "")

        t = str(ref.get("task_type"))
        if t == SubtaskType.MATH.value and isinstance(expected, (int, float)):
            pred = _extract_int(out)
            return {"metric": "accuracy", "score": 1.0 if pred == int(expected) else 0.0}
        if t == SubtaskType.REASONING.value and isinstance(expected, str):
            pred = _norm_yesno(out)
            return {"metric": "accuracy", "score": 1.0 if pred == expected else 0.0}
        if t == SubtaskType.EXTRACTION.value and isinstance(expected, dict):
            try:
                data = json.loads(out.strip())
                ok = all(str(data.get(k)) == str(v) for k, v in expected.items())
                return {"metric": "accuracy", "score": 1.0 if ok else 0.0}
            except Exception:
                return {"metric": "accuracy", "score": 0.0}
        if t == SubtaskType.SUMMARIZE.value and isinstance(expected, dict) and "keywords" in expected:
            text = out.lower()
            ok = all(k.lower() in text for k in expected["keywords"])
            return {"metric": "keywords", "score": 1.0 if ok else 0.0}
        if t == SubtaskType.CODE.value and isinstance(expected, dict):
            code = extract_python_code(out)
            entry = str(expected.get("entry_point") or "")
            tests = expected.get("tests") or []
            tests = [x for x in tests if isinstance(x, str)]
            if code_eval_enabled() and tests:
                harness = "\n".join(tests) + "\n"
                r = run_code_in_subprocess(code, harness, timeout_s=3.0)
                return {"metric": "pass@1", "score": 1.0 if r.passed else 0.0, "error": r.error}
            if entry and f"def {entry}" in code:
                return {"metric": "heuristic_contains_def", "score": 1.0}
            return {"metric": "heuristic_contains_def", "score": 0.0}

        return {"metric": "accuracy", "score": 0.0}


register(CodeMathMix())

