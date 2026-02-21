from __future__ import annotations

import random
import re
from dataclasses import dataclass

from .registry import BenchmarkExample, register


def _extract_final_number(text: str) -> str | None:
    t = (text or "").strip()
    if not t:
        return None
    # Prefer lines containing "final"
    for line in reversed(t.splitlines()):
        if "final" in line.lower():
            m = re.findall(r"-?\d+(?:\.\d+)?", line.replace(",", ""))
            if m:
                return m[-1]
    m = re.findall(r"-?\d+(?:\.\d+)?", t.replace(",", ""))
    return m[-1] if m else None


def _gold_from_answer(ans: str) -> str | None:
    # GSM8K answers often have '#### 42'
    if "####" in ans:
        return ans.split("####")[-1].strip().replace(",", "")
    m = re.findall(r"-?\d+(?:\.\d+)?", ans.replace(",", ""))
    return m[-1] if m else None


@dataclass(frozen=True)
class GSM8K:
    name: str = "gsm8k"

    def load(self, *, data_dir: str, limit: int, seed: int) -> list[BenchmarkExample]:
        from datasets import load_dataset

        ds = load_dataset("gsm8k", "main", split="test")
        rnd = random.Random(seed)
        idxs = list(range(len(ds)))
        rnd.shuffle(idxs)
        idxs = idxs[: max(1, limit)]

        out: list[BenchmarkExample] = []
        for i in idxs:
            ex = ds[int(i)]
            q = ex["question"]
            gold = _gold_from_answer(ex["answer"])
            query = (
                "Solve the problem. Output the final numeric answer clearly.\n"
                'End with a line of the form: "Final: <number>".\n\n'
                f"Problem:\n{q}\n"
            )
            out.append(
                BenchmarkExample(
                    benchmark=self.name,
                    example_id=str(i),
                    query=query,
                    reference={"answer": gold},
                )
            )
        return out

    def score(self, *, example: BenchmarkExample, final_answer: str | None) -> dict:
        pred = _extract_final_number(final_answer or "")
        gold = example.reference.get("answer")
        return {"metric": "accuracy", "score": 1.0 if (pred is not None and gold is not None and pred == gold) else 0.0}


register(GSM8K())

