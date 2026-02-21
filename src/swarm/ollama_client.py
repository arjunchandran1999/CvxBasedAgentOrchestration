from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Optional

import httpx


@dataclass(frozen=True)
class OllamaChatResult:
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: float | None = None
    model: str | None = None


class OllamaClient:
    def __init__(
        self,
        *,
        base_url: str = "http://localhost:11434",
        timeout_s: float = 120.0,
        dry_run: bool = False,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.dry_run = dry_run

    async def chat_raw(
        self,
        *,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.2,
        keep_alive: str = "10m",
    ) -> dict[str, Any]:
        if self.dry_run:
            return self._dry_run_response(model=model, system=system, user=user)

        payload = {
            "model": model,
            "stream": False,
            "keep_alive": keep_alive,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {"temperature": temperature},
        }

        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            resp = await client.post(f"{self.base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
        data["_client_latency_ms"] = (time.perf_counter() - t0) * 1000.0
        return data

    async def chat(
        self,
        *,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.2,
        keep_alive: str = "10m",
    ) -> str:
        data = await self.chat_raw(
            model=model,
            system=system,
            user=user,
            temperature=temperature,
            keep_alive=keep_alive,
        )
        msg = (data.get("message") or {}).get("content")
        if not isinstance(msg, str):
            raise OSError("Ollama response missing message.content")
        return msg

    async def chat_with_usage(
        self,
        *,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.2,
        keep_alive: str = "10m",
    ) -> OllamaChatResult:
        data = await self.chat_raw(
            model=model,
            system=system,
            user=user,
            temperature=temperature,
            keep_alive=keep_alive,
        )
        text = (data.get("message") or {}).get("content") or ""
        prompt_tokens = data.get("prompt_eval_count")
        completion_tokens = data.get("eval_count")
        latency_ms = data.get("_client_latency_ms")
        return OllamaChatResult(
            text=str(text),
            prompt_tokens=int(prompt_tokens) if isinstance(prompt_tokens, int) else None,
            completion_tokens=int(completion_tokens) if isinstance(completion_tokens, int) else None,
            latency_ms=float(latency_ms) if isinstance(latency_ms, (int, float)) else None,
            model=str(data.get("model") or model),
        )

    def _dry_run_response(self, *, model: str, system: str, user: str) -> dict[str, Any]:
        # Minimal deterministic stubs so the full pipeline can run without Ollama.
        content: str
        if "task decomposer" in system.lower():
            # Return a few subtasks depending on keywords.
            q = user.lower()
            subtasks = []
            if "python" in q or "code" in q or "function" in q:
                subtasks.append(
                    {
                        "id": "t1",
                        "task_type": "code",
                        "description": "Implement the requested code artifact.",
                        "estimated_tokens": 800,
                        "difficulty": 3.0,
                    }
                )
                subtasks.append(
                    {
                        "id": "t2",
                        "task_type": "summarize",
                        "description": "Summarize usage and edge cases.",
                        "estimated_tokens": 400,
                        "difficulty": 2.0,
                    }
                )
            elif "solve" in q or "math" in q:
                subtasks.append(
                    {
                        "id": "t1",
                        "task_type": "math",
                        "description": "Solve the math problem with steps.",
                        "estimated_tokens": 700,
                        "difficulty": 3.5,
                    }
                )
                subtasks.append(
                    {
                        "id": "t2",
                        "task_type": "reasoning",
                        "description": "Explain the result in plain language.",
                        "estimated_tokens": 400,
                        "difficulty": 2.0,
                    }
                )
            else:
                subtasks.append(
                    {
                        "id": "t1",
                        "task_type": "reasoning",
                        "description": "Answer the user query thoroughly.",
                        "estimated_tokens": 900,
                        "difficulty": 3.0,
                    }
                )
            content = json.dumps({"subtasks": subtasks})
        else:
            content = "DRY_RUN: " + (user[:200] if user else "")

        # Mimic Ollama /api/chat shape.
        return {
            "model": model,
            "created_at": "dry-run",
            "message": {"role": "assistant", "content": content},
            "done": True,
            "prompt_eval_count": max(1, len(user) // 4),
            "eval_count": max(1, len(content) // 4),
            "_client_latency_ms": 5.0,
        }

