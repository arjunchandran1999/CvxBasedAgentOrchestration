from __future__ import annotations

import random
import re
from dataclasses import dataclass

from .registry import BenchmarkExample, register
from ..agents import SubtaskType
from ..tasks import Subtask


def _normalize_choice(ans: str) -> str | None:
    a = (ans or "").strip().upper()
    if not a:
        return None
    m = re.search(r"\b([A-E])\b", a)
    return m.group(1) if m else None


@dataclass(frozen=True)
class ARCChallenge:
    name: str = "arc_challenge"

    def load(self, *, data_dir: str, limit: int, seed: int) -> list[BenchmarkExample]:
        from datasets import load_dataset

        ds = load_dataset("ai2_arc", "ARC-Challenge", split="test")
        rnd = random.Random(seed)
        idxs = list(range(len(ds)))
        rnd.shuffle(idxs)
        idxs = idxs[: max(1, limit)]

        out: list[BenchmarkExample] = []
        for i in idxs:
            ex = ds[int(i)]
            q = ex["question"]
            choices = ex["choices"]
            labels = choices["label"]
            texts = choices["text"]
            choice_lines = "\n".join([f"{l}. {t}" for l, t in zip(labels, texts)])
            query = (
                "Choose the correct option. Output ONLY the letter (A, B, C, D, or E).\n\n"
                f"Question:\n{q}\n\n"
                f"Choices:\n{choice_lines}\n"
            )
            out.append(
                BenchmarkExample(
                    benchmark=self.name,
                    example_id=str(ex.get("id", i)),
                    query=query,
                    reference={"answer": str(ex["answerKey"]).strip().upper()},
                )
            )
        return out

    def score(self, *, example: BenchmarkExample, final_answer: str | None) -> dict:
        pred = _normalize_choice(final_answer or "")
        gold = str(example.reference.get("answer"))
        return {"metric": "accuracy", "score": 1.0 if pred == gold else 0.0}

    def make_subtasks(self, *, example: BenchmarkExample) -> list[Subtask]:
        return [
            Subtask(
                id="t1",
                task_type=SubtaskType.REASONING,
                description=example.query,
                estimated_tokens=384,
                difficulty=2.5,
            )
        ]


register(ARCChallenge())

