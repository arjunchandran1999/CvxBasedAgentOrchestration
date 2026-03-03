from __future__ import annotations

import random
import re
from dataclasses import dataclass

from ..agents import SubtaskType
from ..tasks import Subtask
from .registry import BenchmarkExample, register


_CHOICE_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)


def _normalize_choice(text: str) -> str | None:
    """
    Normalize model output to a single answer choice letter (A/B/C/D).

    We accept outputs like:
      - "A"
      - "Final: B"
      - "Answer is (c)."
    """
    t = (text or "").strip().upper()
    if not t:
        return None
    m = _CHOICE_RE.search(t)
    return m.group(1).upper() if m else None


@dataclass(frozen=True)
class MMLU:
    """
    MMLU-style multiple-choice benchmark.

    Notes:
    - Uses HF datasets via `datasets`.
    - Default config is `cais/mmlu` with a chosen subject. If that dataset/config
      isn't available, users should adjust or vendor a local question set.
    """

    name: str = "mmlu"
    hf_dataset: str = "cais/mmlu"
    # Default to a small, common subject. Users can swap by passing `--benchmark mmlu:subject`.
    default_subject: str = "high_school_mathematics"

    def load(self, *, data_dir: str, limit: int, seed: int) -> list[BenchmarkExample]:
        from datasets import load_dataset

        # Allow overriding subject via BENCHMARK name convention: "mmlu:<subject>"
        # BenchRunner passes benchmark names directly, so this class remains subject-agnostic.
        subject = self.default_subject
        # If someone registers multiple instances, they can set default_subject differently.

        ds = load_dataset(self.hf_dataset, subject, split="test")
        rnd = random.Random(seed)
        idxs = list(range(len(ds)))
        rnd.shuffle(idxs)
        idxs = idxs[: max(1, limit)]

        out: list[BenchmarkExample] = []
        for i in idxs:
            ex = ds[int(i)]
            question = str(ex.get("question") or "")
            choices = ex.get("choices") or ex.get("options") or None
            answer = ex.get("answer")

            # `cais/mmlu` uses: choices(list[str]) and answer(int index).
            if not isinstance(choices, list) or len(choices) < 4:
                continue
            if isinstance(answer, int):
                gold = "ABCD"[int(answer)] if 0 <= int(answer) < 4 else None
            elif isinstance(answer, str):
                gold = _normalize_choice(answer)
            else:
                gold = None

            choice_lines = "\n".join([f"{lab}. {txt}" for lab, txt in zip(["A", "B", "C", "D"], choices[:4])])
            query = (
                "Answer the multiple-choice question. Output ONLY the letter A, B, C, or D.\n\n"
                f"Question:\n{question}\n\n"
                f"Choices:\n{choice_lines}\n"
            )
            out.append(
                BenchmarkExample(
                    benchmark=self.name,
                    example_id=f"{subject}:{i}",
                    query=query,
                    reference={"answer": gold, "subject": subject},
                )
            )
        return out

    def make_subtasks(self, *, example: BenchmarkExample) -> list[Subtask]:
        return [
            Subtask(
                id="t1",
                task_type=SubtaskType.REASONING,
                description=example.query,
                estimated_tokens=256,
                difficulty=3.0,
            )
        ]

    def score(self, *, example: BenchmarkExample, final_answer: str | None) -> dict:
        pred = _normalize_choice(final_answer or "")
        gold = str(example.reference.get("answer") or "")
        return {"metric": "accuracy", "score": 1.0 if pred and gold and pred == gold else 0.0}


register(MMLU())

