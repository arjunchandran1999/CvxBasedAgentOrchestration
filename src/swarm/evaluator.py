from __future__ import annotations

import json
from dataclasses import dataclass

from pydantic import BaseModel, Field, ValidationError

from .ollama_client import OllamaClient
from .tasks import Subtask
from .executor import SubtaskResult


@dataclass(frozen=True)
class Evaluation:
    subtask_id: str
    success: bool
    judge_score: float | None
    judge_rationale: str | None = None


class _JudgeOut(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    rationale: str = Field(..., min_length=1, max_length=2000)


_JUDGE_SYSTEM = "You are a strict grader. Output ONLY JSON, no markdown."


def _coerce_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1] if "```" in text else text
        text = text.strip()
    return json.loads(text)


async def evaluate(
    *,
    subtasks: list[Subtask],
    results: list[SubtaskResult],
    judge: bool,
    ollama: OllamaClient,
    judge_model: str = "llama3.1:8b",
) -> list[Evaluation]:
    by_id = {s.id: s for s in subtasks}
    evals: list[Evaluation] = []

    for r in results:
        s = by_id.get(r.subtask_id)
        if s is None:
            evals.append(Evaluation(subtask_id=r.subtask_id, success=False, judge_score=None, judge_rationale="unknown subtask"))
            continue

        success = bool(r.success and r.output.strip())
        if not judge:
            evals.append(Evaluation(subtask_id=s.id, success=success, judge_score=None))
            continue

        prompt = {
            "task": {
                "id": s.id,
                "type": s.task_type.value,
                "description": s.description,
                "difficulty": s.difficulty,
            },
            "candidate_output": r.output,
            "scoring": {
                "score": "0..1 (1 is excellent: correct, complete, readable; 0 is wrong/empty)",
                "focus": ["correctness", "completeness", "clarity"],
            },
            "output_schema": {"score": "number 0..1", "rationale": "short string"},
            "rules": ["Return ONLY JSON."],
        }

        try:
            text = await ollama.chat(model=judge_model, system=_JUDGE_SYSTEM, user=json.dumps(prompt, ensure_ascii=False), temperature=0.0)
            out = _JudgeOut.model_validate(_coerce_json(text))
            evals.append(Evaluation(subtask_id=s.id, success=success, judge_score=float(out.score), judge_rationale=out.rationale))
        except (OSError, ValidationError, json.JSONDecodeError):
            evals.append(Evaluation(subtask_id=s.id, success=success, judge_score=None, judge_rationale="judge_failed"))

    return evals

