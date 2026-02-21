from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from ..agents import SubtaskType
from ..task_graph import TaskDAG, TaskNode
from ..tasks import Subtask
from .code_eval import code_eval_enabled, extract_python_code, run_code_in_subprocess
from .registry import BenchmarkExample, register


@dataclass(frozen=True)
class WorkflowSubtaskSpec:
    id: str
    task_type: SubtaskType
    description: str
    expected: Any
    estimated_tokens: int = 400
    difficulty: float = 2.5


@dataclass(frozen=True)
class WorkflowJob:
    job_id: str
    subtasks: list[WorkflowSubtaskSpec]
    # For DAG: list of (parent_id, child_id) per job. wb1: s0->s1, s1->s2; wb2: s0->s1, s0->s2; wb3: s0->s1, s1->s2
    edges: list[tuple[str, str]] | None = None


def _extract_ints(text: str) -> list[int]:
    return [int(x) for x in re.findall(r"-?\d+", text or "")]


def _score_extraction_json(output: str, expected: dict) -> float:
    try:
        data = json.loads(output.strip())
        ok = True
        for k, v in expected.items():
            ok = ok and (str(data.get(k)) == str(v))
        return 1.0 if ok else 0.0
    except Exception:
        # fallback: numeric values by numeric match; strings by substring match
        out = (output or "").lower()
        ints = _extract_ints(output)
        for _, v in expected.items():
            if isinstance(v, (int, float)):
                if int(v) not in ints:
                    return 0.0
            elif isinstance(v, str):
                # numeric-like strings treated numerically; otherwise substring match
                if re.fullmatch(r"-?\d+", v.strip()):
                    if int(v.strip()) not in ints:
                        return 0.0
                else:
                    if v.lower() not in out:
                        return 0.0
            else:
                if str(v).lower() not in out:
                    return 0.0
        return 1.0


def _score_math_numeric(output: str, expected: int | float) -> float:
    ints = _extract_ints(output)
    if not ints:
        return 0.0
    return 1.0 if ints[-1] == int(expected) else 0.0


def _score_contains(output: str, needle: str) -> float:
    return 1.0 if needle.lower() in (output or "").lower() else 0.0


def _score_keywords(output: str, keywords: list[str]) -> float:
    out = (output or "").lower()
    hit = sum(1 for k in keywords if k.lower() in out)
    return float(hit) / float(max(1, len(keywords)))


def _score_code(output: str, entry_point: str, expected_value: Any) -> float:
    code = extract_python_code(output or "")
    if not code:
        return 0.0

    if code_eval_enabled():
        harness = (
            f"def _run():\n"
            f"    v = {entry_point}(3, 4)\n"
            f"    assert v == {expected_value!r}\n"
            f"_run()\n"
        )
        r = run_code_in_subprocess(code, harness, timeout_s=3.0)
        return 1.0 if r.passed else 0.0

    return 1.0 if f"def {entry_point}" in code else 0.5


class WorkflowBench:
    name = "workflowbench"

    def __init__(self) -> None:
        self.jobs: list[WorkflowJob] = [
            WorkflowJob(
                job_id="wb1",
                subtasks=[
                    WorkflowSubtaskSpec(
                        id="s0",
                        task_type=SubtaskType.EXTRACTION,
                        description="From the text 'A=3 and B=4', extract JSON {\"A\":3,\"B\":4}. Return ONLY JSON.",
                        expected={"A": 3, "B": 4},
                        estimated_tokens=200,
                        difficulty=1.5,
                    ),
                    WorkflowSubtaskSpec(
                        id="s1",
                        task_type=SubtaskType.MATH,
                        description="Given A=3 and B=4, compute A^2 + B^2. Output only the number.",
                        expected=25,
                        estimated_tokens=150,
                        difficulty=2.0,
                    ),
                    WorkflowSubtaskSpec(
                        id="s2",
                        task_type=SubtaskType.CODE,
                        description="Write Python: def hyp_sq(a,b): return a*a + b*b. Return ONLY code.",
                        expected={"entry_point": "hyp_sq", "value": 25},
                        estimated_tokens=350,
                        difficulty=2.0,
                    ),
                ],
                edges=[("s0", "s1"), ("s1", "s2")],
            ),
            WorkflowJob(
                job_id="wb2",
                subtasks=[
                    WorkflowSubtaskSpec(
                        id="s0",
                        task_type=SubtaskType.CODE,
                        description="Write Python: def fib(n): return nth Fibonacci with fib(0)=0,fib(1)=1. Return ONLY code.",
                        expected={"entry_point": "fib", "value": 55},
                        estimated_tokens=450,
                        difficulty=3.0,
                    ),
                    WorkflowSubtaskSpec(
                        id="s1",
                        task_type=SubtaskType.MATH,
                        description="Compute fib(10) if fib(0)=0, fib(1)=1. Output only the number.",
                        expected=55,
                        estimated_tokens=120,
                        difficulty=2.0,
                    ),
                    WorkflowSubtaskSpec(
                        id="s2",
                        task_type=SubtaskType.SUMMARIZE,
                        description="In 1-2 sentences, explain how Fibonacci works (recurrence).",
                        expected={"keywords": ["fib", "previous", "sum"]},
                        estimated_tokens=180,
                        difficulty=2.0,
                    ),
                ],
                edges=[("s0", "s1"), ("s0", "s2")],
            ),
            WorkflowJob(
                job_id="wb3",
                subtasks=[
                    WorkflowSubtaskSpec(
                        id="s0",
                        task_type=SubtaskType.EXTRACTION,
                        description="Extract the date from: 'Meeting is on 2026-02-21.' Return ONLY JSON {\"date\": \"YYYY-MM-DD\"}.",
                        expected={"date": "2026-02-21"},
                        estimated_tokens=180,
                        difficulty=1.5,
                    ),
                    WorkflowSubtaskSpec(
                        id="s1",
                        task_type=SubtaskType.REASONING,
                        description="Is 2026-02-21 a Saturday? Answer YES or NO only.",
                        expected="yes",
                        estimated_tokens=120,
                        difficulty=2.5,
                    ),
                    WorkflowSubtaskSpec(
                        id="s2",
                        task_type=SubtaskType.SUMMARIZE,
                        description="Summarize the result in one short sentence.",
                        expected={"keywords": ["saturday"]},
                        estimated_tokens=120,
                        difficulty=1.5,
                    ),
                ],
                edges=[("s0", "s1"), ("s1", "s2")],
            ),
        ]

    def load(self, *, data_dir: str, limit: int, seed: int) -> list[BenchmarkExample]:
        xs = self.jobs[: max(1, min(limit, len(self.jobs)))]
        out = []
        for j in xs:
            out.append(
                BenchmarkExample(
                    benchmark=self.name,
                    example_id=j.job_id,
                    query=f"WorkflowBench job {j.job_id}",
                    reference={"job_id": j.job_id, "subtasks": [self._spec_to_ref(s) for s in j.subtasks]},
                )
            )
        return out

    def _spec_to_ref(self, s: WorkflowSubtaskSpec) -> dict:
        return {
            "id": s.id,
            "task_type": s.task_type.value,
            "description": s.description,
            "expected": s.expected,
            "estimated_tokens": s.estimated_tokens,
            "difficulty": s.difficulty,
        }

    def get_task_dag(self, *, example: BenchmarkExample) -> TaskDAG | None:
        """Return a TaskDAG for workflowbench jobs with dependencies."""
        job_id = str(example.reference.get("job_id") or example.example_id)
        job = next((j for j in self.jobs if j.job_id == job_id), None)
        if job is None or getattr(job, "edges", None) is None:
            return None
        edges = job.edges or []
        parent_map: dict[str, list[str]] = {s.id: [] for s in job.subtasks}
        for p, c in edges:
            if c in parent_map:
                parent_map[c].append(p)
        nodes: dict[str, TaskNode] = {}
        for s in job.subtasks:
            nodes[s.id] = TaskNode(
                id=s.id,
                task_type=s.task_type,
                description=s.description,
                estimated_tokens=s.estimated_tokens,
                difficulty=s.difficulty,
                depends_on=parent_map.get(s.id, []),
            )
        return TaskDAG(job_id=job_id, nodes=nodes)

    def make_subtasks(self, *, example: BenchmarkExample) -> list[Subtask] | None:
        subs = example.reference.get("subtasks") or []
        out: list[Subtask] = []
        for s in subs:
            out.append(
                Subtask(
                    id=str(s["id"]),
                    task_type=SubtaskType(str(s["task_type"])),
                    description=str(s["description"]),
                    estimated_tokens=int(s.get("estimated_tokens", 300)),
                    difficulty=float(s.get("difficulty", 2.5)),
                )
            )
        return out

    def score(self, *, example: BenchmarkExample, final_answer: str | None) -> dict:
        # Not used (workflowbench scores via artifact / subtask outputs)
        return {"metric": "job_score", "score": 0.0}

    def score_artifact(self, *, example: BenchmarkExample, artifact: dict) -> dict:
        ref = example.reference
        subrefs = ref.get("subtasks") or []
        results = {r["subtask_id"]: r for r in (artifact.get("results") or []) if isinstance(r, dict)}

        scores: list[float] = []
        for s in subrefs:
            sid = str(s["id"])
            r = results.get(sid) or {}
            out = str(r.get("output") or "")
            t = str(s.get("task_type"))
            exp = s.get("expected")
            if t == SubtaskType.EXTRACTION.value and isinstance(exp, dict):
                scores.append(_score_extraction_json(out, exp))
            elif t == SubtaskType.MATH.value and isinstance(exp, (int, float)):
                scores.append(_score_math_numeric(out, exp))
            elif t == SubtaskType.CODE.value and isinstance(exp, dict):
                scores.append(_score_code(out, str(exp.get("entry_point")), exp.get("value")))
            elif t == SubtaskType.REASONING.value and isinstance(exp, str):
                scores.append(1.0 if (out.strip().lower().startswith(exp.lower())) else 0.0)
            elif isinstance(exp, dict) and "keywords" in exp:
                scores.append(_score_keywords(out, list(exp["keywords"])))
            elif isinstance(exp, str):
                scores.append(_score_contains(out, exp))
            else:
                scores.append(0.0)

        job_score = float(sum(scores) / len(scores)) if scores else 0.0
        return {"metric": "job_score", "score": job_score, "subtask_scores": scores}


register(WorkflowBench())

