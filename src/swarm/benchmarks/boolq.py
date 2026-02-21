from __future__ import annotations

import random
import re
from dataclasses import dataclass

from .registry import BenchmarkExample, register


def _normalize_bool(ans: str) -> str | None:
    a = ans.strip().lower()
    if not a:
        return None
    if a.startswith("yes") or a == "true":
        return "yes"
    if a.startswith("no") or a == "false":
        return "no"
    # sometimes "Yes." etc
    m = re.match(r"^(yes|no)\b", a)
    return m.group(1) if m else None


@dataclass(frozen=True)
class BoolQ:
    name: str = "boolq"

    def load(self, *, data_dir: str, limit: int, seed: int) -> list[BenchmarkExample]:
        from datasets import load_dataset  # lazy import

        ds = load_dataset("super_glue", "boolq", split="validation")
        rnd = random.Random(seed)
        idxs = list(range(len(ds)))
        rnd.shuffle(idxs)
        idxs = idxs[: max(1, limit)]

        out: list[BenchmarkExample] = []
        for i in idxs:
            ex = ds[int(i)]
            passage = ex["passage"]
            question = ex["question"]
            label = bool(ex["label"])
            query = (
                "Answer YES or NO only.\n\n"
                f"Passage:\n{passage}\n\n"
                f"Question:\n{question}\n"
            )
            out.append(
                BenchmarkExample(
                    benchmark=self.name,
                    example_id=str(ex.get("idx", i)),
                    query=query,
                    reference={"label": "yes" if label else "no"},
                )
            )
        return out

    def score(self, *, example: BenchmarkExample, final_answer: str | None) -> dict:
        pred = _normalize_bool(final_answer or "")
        gold = str(example.reference.get("label"))
        return {"metric": "accuracy", "score": 1.0 if pred == gold else 0.0}


register(BoolQ())

