from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Literal


Label = Literal["a", "b", "tie"]


@dataclass(frozen=True)
class PreferenceExample:
    prompt: str
    answer_a: str
    answer_b: str
    label: Label  # which answer is better
    meta: dict


JudgeFn = Callable[[str, str, str], Label]


def write_preferences_jsonl(path: Path, rows: list[PreferenceExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


def read_preferences_jsonl(path: Path) -> list[PreferenceExample]:
    out: list[PreferenceExample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        out.append(
            PreferenceExample(
                prompt=str(d["prompt"]),
                answer_a=str(d["answer_a"]),
                answer_b=str(d["answer_b"]),
                label=str(d["label"]),  # type: ignore[arg-type]
                meta=dict(d.get("meta") or {}),
            )
        )
    return out

