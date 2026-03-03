from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..agents import SubtaskType
from ..tasks import Subtask
from .registry import BenchmarkExample, register


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _mtbench_questions_path() -> Path:
    return _repo_root() / "data" / "mt_bench" / "questions.jsonl"


def _coerce_question(line: str) -> dict[str, Any] | None:
    try:
        d = json.loads(line)
    except Exception:
        return None
    if not isinstance(d, dict):
        return None
    qid = d.get("question_id") or d.get("id") or d.get("question") or None
    turns = d.get("turns") or d.get("messages") or None
    if not isinstance(qid, str):
        return None
    if isinstance(turns, list):
        turns = [str(t) for t in turns if str(t).strip()]
    else:
        turns = None
    if not turns:
        return None
    return {"question_id": qid, "category": str(d.get("category") or "unknown"), "turns": turns}


def _format_turns(turns: list[str]) -> str:
    # Simple serialization: we model this as a single-shot task with multi-turn content.
    # Real MT-Bench uses turn-by-turn generation; we keep this simple and deterministic for routing evaluation.
    parts = ["You are chatting with a user. Respond to each user turn in order.\n"]
    for i, t in enumerate(turns, start=1):
        parts.append(f"User turn {i}:\n{t}\n")
    parts.append("Return your responses. If multiple turns, separate them clearly.\n")
    return "\n".join(parts)


@dataclass(frozen=True)
class MTBench:
    name: str = "mt_bench"

    def load(self, *, data_dir: str, limit: int, seed: int) -> list[BenchmarkExample]:
        path = _mtbench_questions_path()
        if not path.exists():
            raise FileNotFoundError(
                f"MT-Bench questions not found at {path}. See data/mt_bench/README.md for instructions."
            )
        raw = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        qs = []
        for ln in raw:
            q = _coerce_question(ln)
            if q:
                qs.append(q)
        if not qs:
            raise ValueError(f"No valid questions found in {path}")

        rnd = random.Random(seed)
        rnd.shuffle(qs)
        qs = qs[: max(1, min(limit, len(qs)))]

        out: list[BenchmarkExample] = []
        for q in qs:
            turns = list(q["turns"])
            query = _format_turns(turns)
            out.append(
                BenchmarkExample(
                    benchmark=self.name,
                    example_id=str(q["question_id"]),
                    query=query,
                    reference={"category": q.get("category", "unknown"), "turns": turns},
                )
            )
        return out

    def make_subtasks(self, *, example: BenchmarkExample) -> list[Subtask]:
        return [
            Subtask(
                id="t1",
                task_type=SubtaskType.REASONING,
                description=example.query,
                estimated_tokens=800,
                difficulty=3.0,
            )
        ]

    def score(self, *, example: BenchmarkExample, final_answer: str | None) -> dict:
        # MT-Bench is normally judged pairwise. We keep `score()` as a placeholder so the bench runner
        # can run end-to-end and produce artifacts; pairwise judging is done separately (or via gated reporting).
        _ = (example, final_answer)
        return {"metric": "pairwise_pending", "score": 0.0}


register(MTBench())

