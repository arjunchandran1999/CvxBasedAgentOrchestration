from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from ..mtbench_judge import judge_pairwise_sync
from ..benchmarks.registry import ensure_default_benchmarks_loaded, get as get_benchmark, BenchmarkExample
from .preference_data import PreferenceExample


def _ollama_chat_sync(*, base_url: str, model: str, user: str, system: str | None = None, timeout_s: float = 120.0) -> str:
    url = base_url.rstrip("/") + "/api/chat"
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    req = {"model": model, "messages": messages, "stream": False, "options": {"temperature": 0.2}}
    with httpx.Client(timeout=timeout_s) as client:
        r = client.post(url, json=req)
        r.raise_for_status()
        data = r.json()
    if isinstance(data, dict) and isinstance(data.get("message"), dict):
        return str(data["message"].get("content") or "")
    return ""


def generate_preferences_from_benchmark(
    *,
    benchmark: str,
    data_dir: str,
    limit: int,
    seed: int,
    strong_model: str,
    weak_model: str,
    judge_model: str,
    ollama_base_url: str = "http://localhost:11434",
    timeout_s: float = 120.0,
) -> list[PreferenceExample]:
    """
    Generate in-domain preference labels by comparing strong vs weak outputs on benchmark prompts,
    judged by a local judge model.\n+
    Label semantics:
      - label == "a"  => answer_a (strong) wins
      - label == "b"  => answer_b (weak) wins
      - label == "tie"
    """
    ensure_default_benchmarks_loaded()
    bench = get_benchmark(benchmark)
    examples = bench.load(data_dir=data_dir, limit=limit, seed=seed)

    rows: list[PreferenceExample] = []
    for ex in examples:
        prompt = str(ex.query)
        ans_strong = _ollama_chat_sync(base_url=ollama_base_url, model=strong_model, user=prompt, timeout_s=timeout_s)
        ans_weak = _ollama_chat_sync(base_url=ollama_base_url, model=weak_model, user=prompt, timeout_s=timeout_s)
        res = judge_pairwise_sync(
            base_url=ollama_base_url,
            model=judge_model,
            prompt=prompt,
            answer_a=ans_strong,
            answer_b=ans_weak,
            timeout_s=timeout_s,
        )
        rows.append(
            PreferenceExample(
                prompt=prompt,
                answer_a=ans_strong,
                answer_b=ans_weak,
                label=res.winner,
                meta={"benchmark": benchmark, "example_id": ex.example_id, "seed": seed},
            )
        )
    return rows


def write_preferences(path: Path, rows: list[PreferenceExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"prompt": r.prompt, "answer_a": r.answer_a, "answer_b": r.answer_b, "label": r.label, "meta": r.meta}, ensure_ascii=False) + "\n")

