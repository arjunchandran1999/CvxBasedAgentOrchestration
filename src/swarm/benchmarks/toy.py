from __future__ import annotations

import random
from dataclasses import dataclass

from .registry import BenchmarkExample, register


@dataclass(frozen=True)
class ToyBenchmark:
    name: str = "toy"

    def load(self, *, data_dir: str, limit: int, seed: int) -> list[BenchmarkExample]:
        rnd = random.Random(seed)
        examples: list[BenchmarkExample] = []
        for i in range(max(1, limit)):
            kind = rnd.choice(["math", "code", "summarize"])
            if kind == "math":
                a, b = rnd.randint(10, 99), rnd.randint(10, 99)
                query = f"Solve: {a} + {b}. Output only the number."
                ref = {"answer": str(a + b), "type": "math"}
            elif kind == "code":
                query = "Write a Python function `add(a,b)` that returns a+b."
                ref = {"must_contain": "def add", "type": "code"}
            else:
                query = "Summarize: Cats are mammals. They purr. They like naps."
                ref = {"must_contain_any": ["mammals", "purr", "naps"], "type": "summarize"}
            examples.append(BenchmarkExample(benchmark=self.name, example_id=f"{self.name}-{i+1}", query=query, reference=ref))
        return examples

    def score(self, *, example: BenchmarkExample, final_answer: str | None) -> dict:
        ans = (final_answer or "").strip().lower()
        ref = example.reference
        if ref.get("type") == "math":
            ok = ans.split()[0] == str(ref.get("answer"))
            return {"metric": "accuracy", "score": 1.0 if ok else 0.0}
        if ref.get("type") == "code":
            ok = "def add" in ans
            return {"metric": "contains", "score": 1.0 if ok else 0.0}
        must = ref.get("must_contain_any") or []
        ok = any(m in ans for m in must)
        return {"metric": "contains_any", "score": 1.0 if ok else 0.0}


register(ToyBenchmark())

