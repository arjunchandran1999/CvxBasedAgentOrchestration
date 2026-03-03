from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True)
class PairwiseResult:
    winner: str  # "a" | "b" | "tie"
    rationale: str | None = None


_SYSTEM = "You are a strict judge. Output ONLY valid JSON."


def _coerce_json(text: str) -> Any:
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1] if "```" in t else t
        t = t.strip()
    return json.loads(t)


def judge_pairwise_sync(
    *,
    base_url: str,
    model: str,
    prompt: str,
    answer_a: str,
    answer_b: str,
    timeout_s: float = 120.0,
) -> PairwiseResult:
    """
    Minimal MT-Bench-style pairwise judge using local Ollama (sync).
    Returns winner in {a,b,tie}.
    """
    payload = {
        "prompt": prompt,
        "answer_a": answer_a,
        "answer_b": answer_b,
        "rubric": ["helpfulness", "correctness", "completeness", "clarity"],
        "output_schema": {"winner": "a|b|tie", "rationale": "short string"},
        "rules": ["Return ONLY JSON.", "Pick tie only if truly indistinguishable."],
    }
    req = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        "stream": False,
        "options": {"temperature": 0.0},
    }

    url = base_url.rstrip("/") + "/api/chat"
    with httpx.Client(timeout=timeout_s) as client:
        resp = client.post(url, json=req)
        resp.raise_for_status()
        data = resp.json()

    text = ""
    if isinstance(data, dict):
        msg = data.get("message") or {}
        if isinstance(msg, dict):
            text = str(msg.get("content") or "")
    out = _coerce_json(text)
    if not isinstance(out, dict):
        return PairwiseResult(winner="tie", rationale="judge_parse_failed")
    w = str(out.get("winner") or "tie").strip().lower()
    if w not in {"a", "b", "tie"}:
        w = "tie"
    rat = out.get("rationale")
    return PairwiseResult(winner=w, rationale=str(rat) if isinstance(rat, str) else None)

