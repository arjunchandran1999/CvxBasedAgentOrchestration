from __future__ import annotations

import random
from dataclasses import dataclass

from rouge_score import rouge_scorer

from .registry import BenchmarkExample, register
from ..agents import SubtaskType
from ..tasks import Subtask


_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


@dataclass(frozen=True)
class XSum:
    name: str = "xsum"

    def load(self, *, data_dir: str, limit: int, seed: int) -> list[BenchmarkExample]:
        from datasets import load_dataset

        ds = load_dataset("xsum", split="test")
        rnd = random.Random(seed)
        idxs = list(range(len(ds)))
        rnd.shuffle(idxs)
        idxs = idxs[: max(1, limit)]

        out: list[BenchmarkExample] = []
        for i in idxs:
            ex = ds[int(i)]
            doc = ex["document"]
            summary = ex["summary"]
            query = (
                "Write a one-sentence summary of the following document.\n\n"
                f"Document:\n{doc}\n"
            )
            out.append(
                BenchmarkExample(
                    benchmark=self.name,
                    example_id=str(ex.get("id", i)),
                    query=query,
                    reference={"summary": summary},
                )
            )
        return out

    def score(self, *, example: BenchmarkExample, final_answer: str | None) -> dict:
        pred = (final_answer or "").strip()
        ref = str(example.reference.get("summary") or "")
        if not pred or not ref:
            return {"metric": "rougeL", "score": 0.0}
        s = _scorer.score(ref, pred)["rougeL"].fmeasure
        return {"metric": "rougeL", "score": float(s)}

    def make_subtasks(self, *, example: BenchmarkExample) -> list[Subtask]:
        return [
            Subtask(
                id="t1",
                task_type=SubtaskType.SUMMARIZE,
                description=example.query,
                estimated_tokens=200,
                difficulty=2.0,
            )
        ]


register(XSum())

