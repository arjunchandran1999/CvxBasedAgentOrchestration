"""Agentic benchmark: DAG jobs with planner/coder/tester/summarizer roles."""

from __future__ import annotations

from dataclasses import dataclass

from ..agents import SubtaskType
from ..task_graph import TaskDAG, TaskNode
from ..tasks import Subtask
from .registry import BenchmarkExample, register

# Reuse workflowbench scoring helpers
from .workflowbench import _score_extraction_json, _score_math_numeric, _score_code, _score_keywords


@dataclass(frozen=True)
class AgenticSubtaskSpec:
    id: str
    task_type: SubtaskType
    description: str
    expected: object
    estimated_tokens: int = 300
    difficulty: float = 2.5


@dataclass(frozen=True)
class AgenticJob:
    job_id: str
    subtasks: list[AgenticSubtaskSpec]
    edges: list[tuple[str, str]]


class AgenticBench:
    """Complex agentic benchmark: planner → coder → tester → summarizer."""

    name = "agentic_bench"

    def __init__(self) -> None:
        self.jobs: list[AgenticJob] = [
            AgenticJob(
                job_id="ab1",
                subtasks=[
                    AgenticSubtaskSpec(
                        id="s0",
                        task_type=SubtaskType.REASONING,
                        description="Plan: We need a function that computes factorial of n. Return ONLY a one-line plan.",
                        expected={"keywords": ["factorial", "loop", "multiply"]},
                        estimated_tokens=150,
                        difficulty=2.0,
                    ),
                    AgenticSubtaskSpec(
                        id="s1",
                        task_type=SubtaskType.CODE,
                        description="Write Python: def fact(n): return n! (factorial). Return ONLY code.",
                        expected={"entry_point": "fact", "value": 120},
                        estimated_tokens=200,
                        difficulty=2.5,
                    ),
                    AgenticSubtaskSpec(
                        id="s2",
                        task_type=SubtaskType.MATH,
                        description="Verify: compute fact(5) and output only the number.",
                        expected=120,
                        estimated_tokens=80,
                        difficulty=1.5,
                    ),
                    AgenticSubtaskSpec(
                        id="s3",
                        task_type=SubtaskType.SUMMARIZE,
                        description="Summarize what the factorial function does in one sentence.",
                        expected={"keywords": ["factorial", "product"]},
                        estimated_tokens=80,
                        difficulty=1.5,
                    ),
                ],
                edges=[("s0", "s1"), ("s1", "s2"), ("s2", "s3")],
            ),
        ]

    def load(self, *, data_dir: str, limit: int, seed: int) -> list[BenchmarkExample]:
        xs = self.jobs[: max(1, min(limit, len(self.jobs)))]
        return [
            BenchmarkExample(
                benchmark=self.name,
                example_id=j.job_id,
                query=f"AgenticBench job {j.job_id}",
                reference={"job_id": j.job_id, "subtasks": [self._spec_to_ref(s) for s in j.subtasks]},
            )
            for j in xs
        ]

    def _spec_to_ref(self, s: AgenticSubtaskSpec) -> dict:
        return {
            "id": s.id,
            "task_type": s.task_type.value,
            "description": s.description,
            "expected": s.expected,
            "estimated_tokens": s.estimated_tokens,
            "difficulty": s.difficulty,
        }

    def get_task_dag(self, *, example: BenchmarkExample) -> TaskDAG | None:
        job_id = str(example.reference.get("job_id") or example.example_id)
        job = next((j for j in self.jobs if j.job_id == job_id), None)
        if not job:
            return None
        parent_map: dict[str, list[str]] = {s.id: [] for s in job.subtasks}
        for p, c in job.edges:
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
        return [
            Subtask(
                id=str(s["id"]),
                task_type=SubtaskType(str(s["task_type"])),
                description=str(s["description"]),
                estimated_tokens=int(s.get("estimated_tokens", 300)),
                difficulty=float(s.get("difficulty", 2.5)),
            )
            for s in subs
        ]

    def score(self, *, example: BenchmarkExample, final_answer: str | None) -> dict:
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
            if t == SubtaskType.MATH.value and isinstance(exp, (int, float)):
                scores.append(_score_math_numeric(out, exp))
            elif t == SubtaskType.CODE.value and isinstance(exp, dict):
                scores.append(_score_code(out, str(exp.get("entry_point")), exp.get("value")))
            elif isinstance(exp, dict) and "keywords" in exp:
                scores.append(_score_keywords(out, list(exp["keywords"])))
            elif isinstance(exp, dict):
                scores.append(_score_extraction_json(out, exp))
            else:
                scores.append(0.0)

        job_score = float(sum(scores) / len(scores)) if scores else 0.0
        return {"metric": "job_score", "score": job_score, "subtask_scores": scores}


register(AgenticBench())
