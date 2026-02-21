from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from .agents import Agent
from .estimator import QualityEstimator, estimate_token_cost, normalized_switch_cost
from .ollama_client import OllamaClient
from .routing import Assignment, RoutingResult
from .tasks import Subtask


class _AssignmentOut(BaseModel):
    subtask_id: str
    agent: str


class _PlannerOut(BaseModel):
    assignments: list[_AssignmentOut]
    active_models: list[str] | None = None


_SYSTEM = "You are a routing planner. Output ONLY valid JSON."


def _coerce_json(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1] if "```" in text else text
        text = text.strip()
    return json.loads(text)


def _vram_used(active_models: set[str], agents: list[Agent]) -> float:
    by = {a.name: a for a in agents}
    return float(sum(by[m].vram_gb for m in active_models if m in by))


async def llm_assign(
    subtasks: list[Subtask],
    agents: list[Agent],
    estimator: QualityEstimator,
    *,
    ollama: OllamaClient,
    planner_model: str = "llama3.1:8b",
    gpu_vram_gb: float,
    loaded_models: set[str],
    lambda_token: float = 0.5,
    lambda_switch: float = 0.2,
    token_scale: float = 1000.0,
    switch_t_scale_ms: float = 1500.0,
    reprompt_on_violation: bool = True,
    planner_timeout_s: float = 30.0,
) -> RoutingResult:
    if not subtasks:
        return RoutingResult(assignments=[], active_models=[], vram_used_gb=0.0, vram_violation=False, routing_source="llm_planner")

    estimator.ensure_priors(agents)
    agent_names = [a.name for a in agents]
    by_name = {a.name: a for a in agents}

    subtasks_payload = [
        {
            "id": s.id,
            "task_type": s.task_type.value,
            "difficulty": s.difficulty,
            "estimated_tokens": s.estimated_tokens,
            "description": s.description,
        }
        for s in subtasks
    ]
    agents_payload = []
    for a in agents:
        agents_payload.append(
            {
                "name": a.name,
                "vram_gb": a.vram_gb,
                "load_time_ms": a.load_time_ms,
                "cost_per_token": a.cost_per_token,
                "capabilities": {k.value: float(v) for k, v in a.capabilities.items()},
                "loaded": a.name in loaded_models,
            }
        )

    prompt = {
        "goal": "Assign exactly one agent to each subtask. Prefer high quality, low cost, and fewer model loads.",
        "gpu_vram_gb": gpu_vram_gb,
        "constraints": [
            "Return ONLY JSON.",
            "Every subtask_id must appear exactly once.",
            "Use only the provided agent names.",
            "Total VRAM of UNIQUE activated models must be <= gpu_vram_gb.",
        ],
        "loaded_models": sorted(loaded_models),
        "lambda_token": lambda_token,
        "lambda_switch": lambda_switch,
        "token_scale": token_scale,
        "subtasks": subtasks_payload,
        "agents": agents_payload,
        "output_format": {
            "assignments": [{"subtask_id": "string", "agent": "one of agent names"}],
        },
    }

    async def _ask(extra: str | None = None) -> _PlannerOut:
        user = json.dumps(prompt if extra is None else {**prompt, "violation_fix": extra}, ensure_ascii=False)
        text = await ollama.chat(model=planner_model, system=_SYSTEM, user=user, temperature=0.2)
        return _PlannerOut.model_validate(_coerce_json(text))

    try:
        out = await asyncio.wait_for(_ask(), timeout=float(planner_timeout_s))
    except (asyncio.TimeoutError, OSError, httpx.HTTPError, ValidationError, json.JSONDecodeError):
        return _fallback(subtasks, agents, estimator, gpu_vram_gb, loaded_models, lambda_token, lambda_switch, token_scale, switch_t_scale_ms)

    # Validate assignment coverage & agent names.
    seen = set()
    assignments_raw: list[tuple[Subtask, Agent]] = []
    for aout in out.assignments:
        if aout.subtask_id not in {s.id for s in subtasks}:
            continue
        if aout.agent not in agent_names:
            continue
        if aout.subtask_id in seen:
            continue
        seen.add(aout.subtask_id)
        subtask = next(s for s in subtasks if s.id == aout.subtask_id)
        assignments_raw.append((subtask, by_name[aout.agent]))

    # Fill any missing with greedy best among all agents.
    missing = [s for s in subtasks if s.id not in seen]
    for s in missing:
        best = max(agents, key=lambda ag: estimator.predict(ag, s))
        assignments_raw.append((s, best))

    active_models = set(out.active_models or [ag.name for _, ag in assignments_raw])
    active_models |= {ag.name for _, ag in assignments_raw}

    vram_used = _vram_used(active_models, agents)
    if vram_used > gpu_vram_gb + 1e-6 and reprompt_on_violation:
        try:
            out2 = await asyncio.wait_for(
                _ask(
                extra=f"Your selection uses {vram_used:.2f}GB VRAM > {gpu_vram_gb:.2f}GB. "
                "Revise assignments to use fewer/smaller unique models, preferring loaded_models."
                ),
                timeout=float(planner_timeout_s),
            )
            active_models2 = set(out2.active_models or [a.agent for a in out2.assignments])
            active_models2 |= {a.agent for a in out2.assignments}
            vram_used2 = _vram_used(active_models2, agents)
            if vram_used2 <= gpu_vram_gb + 1e-6:
                out = out2
                active_models = active_models2
                vram_used = vram_used2
        except Exception:
            pass

    # Build final Assignment objects.
    assignments: list[Assignment] = []
    for s, ag in assignments_raw:
        p = estimator.predict(ag, s)
        tok = estimate_token_cost(ag, s)
        sw = 0.0 if ag.name in loaded_models else normalized_switch_cost(ag, t_scale_ms=switch_t_scale_ms)
        assignments.append(
            Assignment(
                subtask_id=s.id,
                agent=ag.name,
                mode="llm",
                estimated_performance=float(p),
                estimated_token_cost=float(tok),
                estimated_switch_cost=float(sw),
            )
        )

    assigned_set = {a.agent for a in assignments}
    used_vram = _vram_used(assigned_set, agents)
    vram_violation = used_vram > gpu_vram_gb + 1e-6

    # Report active_models as the actually-used set (even if it violates); we log the violation.
    return RoutingResult(
        assignments=assignments,
        active_models=sorted(assigned_set),
        vram_used_gb=float(used_vram),
        vram_violation=bool(vram_violation),
        routing_source="llm_planner",
    )


def _fallback(
    subtasks: list[Subtask],
    agents: list[Agent],
    estimator: QualityEstimator,
    gpu_vram_gb: float,
    loaded_models: set[str],
    lambda_token: float,
    lambda_switch: float,
    token_scale: float,
    switch_t_scale_ms: float,
) -> RoutingResult:
    # Pick one "good enough" model set within VRAM (prefer loaded; otherwise smallest).
    estimator.ensure_priors(agents)
    candidates = [a for a in agents if a.name in loaded_models]
    if not candidates:
        candidates = sorted(agents, key=lambda x: x.vram_gb)
    active: list[Agent] = []
    for a in candidates:
        if sum(m.vram_gb for m in active) + a.vram_gb <= gpu_vram_gb:
            active.append(a)
    if not active:
        active = [min(agents, key=lambda x: x.vram_gb)]

    assignments: list[Assignment] = []
    for s in subtasks:
        best = None
        for ag in active:
            p = estimator.predict(ag, s)
            tok = estimate_token_cost(ag, s)
            sw = 0.0 if ag.name in loaded_models else normalized_switch_cost(ag, t_scale_ms=switch_t_scale_ms)
            util = p - lambda_token * (tok / token_scale) - lambda_switch * sw
            if best is None or util > best[0]:
                best = (util, ag, p, tok, sw)
        assert best is not None
        _, ag, p, tok, sw = best
        assignments.append(
            Assignment(
                subtask_id=s.id,
                agent=ag.name,
                mode="fallback",
                estimated_performance=float(p),
                estimated_token_cost=float(tok),
                estimated_switch_cost=float(sw),
            )
        )

    active_models = sorted({a.agent for a in assignments})
    vram_used = _vram_used(set(active_models), agents)
    vram_violation = vram_used > gpu_vram_gb + 1e-6
    return RoutingResult(
        assignments=assignments,
        active_models=active_models,
        vram_used_gb=float(vram_used),
        vram_violation=bool(vram_violation),
        routing_source="fallback",
    )

