from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx
from pydantic import BaseModel, Field, ValidationError

from .agents import SubtaskType
from .ollama_client import OllamaClient


@dataclass(frozen=True)
class Subtask:
    id: str
    task_type: SubtaskType
    description: str
    estimated_tokens: int
    difficulty: float  # 1..5-ish


class _SubtaskOut(BaseModel):
    id: str
    task_type: SubtaskType = Field(..., description="One of: code, reasoning, math, creative, extraction, summarize")
    description: str
    estimated_tokens: int = Field(..., ge=1, le=20000)
    difficulty: float = Field(..., ge=0.5, le=10.0)


class _DecomposeOut(BaseModel):
    subtasks: list[_SubtaskOut]


_DECOMPOSE_SYSTEM = (
    "You are a task decomposer. Output ONLY valid JSON, no markdown, no prose."
)


def _fallback_subtask(query: str) -> list[Subtask]:
    return [
        Subtask(
            id="t1",
            task_type=SubtaskType.REASONING,
            description=query,
            estimated_tokens=max(300, min(1200, len(query) // 4 + 300)),
            difficulty=2.5,
        )
    ]


def _coerce_json(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        # Best-effort: strip code fences.
        text = text.split("```", 2)[1] if "```" in text else text
        text = text.strip()
    return json.loads(text)


async def decompose(query: str, ollama: OllamaClient, *, planner_model: str = "llama3.1:8b") -> list[Subtask]:
    prompt = {
        "query": query,
        "output_schema": {
            "subtasks": [
                {
                    "id": "string (short, unique)",
                    "task_type": "code|reasoning|math|creative|extraction|summarize",
                    "description": "string",
                    "estimated_tokens": "integer (rough estimate)",
                    "difficulty": "number (1 easy .. 5 hard)",
                }
            ]
        },
        "rules": [
            "Return ONLY JSON.",
            "Use 1-6 subtasks.",
            "Prefer fewer subtasks for short queries.",
            "Estimated tokens should include expected output.",
        ],
    }

    try:
        text = await ollama.chat(
            model=planner_model,
            system=_DECOMPOSE_SYSTEM,
            user=json.dumps(prompt, ensure_ascii=False),
            temperature=0.2,
        )
        parsed = _DecomposeOut.model_validate(_coerce_json(text))
        subtasks: list[Subtask] = []
        for s in parsed.subtasks:
            subtasks.append(
                Subtask(
                    id=s.id,
                    task_type=s.task_type,
                    description=s.description,
                    estimated_tokens=int(s.estimated_tokens),
                    difficulty=float(s.difficulty),
                )
            )
        if not subtasks:
            return _fallback_subtask(query)
        return subtasks
    except (OSError, httpx.HTTPError, json.JSONDecodeError, ValidationError):
        return _fallback_subtask(query)

