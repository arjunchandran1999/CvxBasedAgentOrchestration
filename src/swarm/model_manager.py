from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Iterable

import httpx


@dataclass(frozen=True)
class OllamaModel:
    name: str
    size_bytes: int | None = None
    digest: str | None = None
    modified_at: str | None = None


class OllamaModelManager:
    def __init__(self, *, base_url: str = "http://localhost:11434", timeout_s: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)

    async def list_models(self) -> list[OllamaModel]:
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            resp = await client.get(f"{self.base_url}/api/tags")
            resp.raise_for_status()
            data = resp.json()

        models: list[OllamaModel] = []
        for m in (data.get("models") or []):
            if not isinstance(m, dict):
                continue
            models.append(
                OllamaModel(
                    name=str(m.get("name") or ""),
                    size_bytes=int(m["size"]) if isinstance(m.get("size"), int) else None,
                    digest=str(m.get("digest")) if m.get("digest") else None,
                    modified_at=str(m.get("modified_at")) if m.get("modified_at") else None,
                )
            )
        return [m for m in models if m.name]

    async def has_model(self, name: str) -> bool:
        name = name.strip()
        if not name:
            return False
        return any(m.name == name for m in await self.list_models())

    async def missing_models(self, names: Iterable[str]) -> list[str]:
        have = {m.name for m in await self.list_models()}
        missing = []
        for n in names:
            nn = str(n).strip()
            if nn and nn not in have:
                missing.append(nn)
        return missing

    def pull_models_cli(self, names: Iterable[str]) -> None:
        """
        Pull models using the `ollama` CLI. This provides good progress output and
        works well for large models. Raises on failure.
        """
        for n in names:
            nn = str(n).strip()
            if not nn:
                continue
            subprocess.check_call(["ollama", "pull", nn])


def normalize_required_models(agent_names: Iterable[str], *, extra: Iterable[str] | None = None) -> list[str]:
    out = []
    seen = set()
    for n in list(agent_names) + list(extra or []):
        nn = str(n).strip()
        if nn and nn not in seen:
            seen.add(nn)
            out.append(nn)
    return out

