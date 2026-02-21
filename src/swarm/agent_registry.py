from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .agents import Agent, SubtaskType, default_agents


def agents_to_jsonable(agents: list[Agent]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for a in agents:
        out.append(
            {
                "name": a.name,
                "cost_per_token": a.cost_per_token,
                "max_ctx": a.max_ctx,
                "capabilities": {k.value: float(v) for k, v in a.capabilities.items()},
                "vram_gb": a.vram_gb,
                "load_time_ms": a.load_time_ms,
            }
        )
    return out


def save_agents_json(agents: list[Agent], path: Path) -> None:
    path.write_text(json.dumps(agents_to_jsonable(agents), indent=2), encoding="utf-8")


def save_default_agents_json(path: Path) -> None:
    save_agents_json(default_agents(), path)


def load_agents(path: Path | None) -> list[Agent]:
    if path is None:
        return default_agents()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("agents file must be a JSON list of agent objects")

    agents: list[Agent] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        caps_raw = item.get("capabilities") or {}
        caps: dict[SubtaskType, float] = {}
        if isinstance(caps_raw, dict):
            for k, v in caps_raw.items():
                try:
                    st = SubtaskType(str(k))
                    caps[st] = float(v)
                except Exception:
                    continue

        agents.append(
            Agent(
                name=str(item["name"]),
                cost_per_token=float(item.get("cost_per_token", 1.0)),
                max_ctx=int(item.get("max_ctx", 8192)),
                capabilities=caps,
                vram_gb=float(item.get("vram_gb", 0.0)),
                load_time_ms=float(item.get("load_time_ms", 0.0)),
            )
        )

    if not agents:
        raise ValueError("agents file produced empty agent list")
    return agents

