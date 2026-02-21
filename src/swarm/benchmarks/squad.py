from __future__ import annotations

import random
import re
from dataclasses import dataclass

from .registry import BenchmarkExample, register


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


@dataclass(frozen=True)
class SQuAD:
    name: str = "squad"

    def load(self, *, data_dir: str, limit: int, seed: int) -> list[BenchmarkExample]:
        from datasets import load_dataset

        ds = load_dataset("squad", split="validation")
        rnd = random.Random(seed)
        idxs = list(range(len(ds)))
        rnd.shuffle(idxs)
        idxs = idxs[: max(1, limit)]

        out: list[BenchmarkExample] = []
        for i in idxs:
            ex = ds[int(i)]
            context = ex["context"]
            question = ex["question"]
            gold = ex["answers"]["text"][0] if ex["answers"]["text"] else ""
            query = (
                "Answer the question using ONLY the provided context. Output ONLY the answer span.\n\n"
                f"Context:\n{context}\n\n"
                f"Question:\n{question}\n"
            )
            out.append(
                BenchmarkExample(
                    benchmark=self.name,
                    example_id=str(ex.get("id", i)),
                    query=query,
                    reference={"answers": ex["answers"]["text"], "gold": gold},
                )
            )
        return out

    def score(self, *, example: BenchmarkExample, final_answer: str | None) -> dict:
        pred = _normalize(final_answer or "")
        answers = example.reference.get("answers") or []
        golds = [_normalize(a) for a in answers if isinstance(a, str)]
        em = 1.0 if pred and pred in set(golds) else 0.0
        return {"metric": "exact_match", "score": em}


register(SQuAD())

