from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class JobTelemetry:
    event: str  # "job"
    ts_ms: float
    run_id: str
    job_id: str
    query: str
    gpu_name: str | None
    gpu_uuid: str | None
    benchmark_name: str | None
    routing_mode: str  # "lp" | "llm"
    lambda_token: float
    lambda_switch: float
    gpu_vram_gb: float
    vram_used_gb: float
    vram_violation: bool
    loaded_models_before: list[str]
    active_models: list[str]
    active_roles: list[str]
    active_role_agents: list[str]  # unique set of "<role>|<model>"
    models_swapped_in: list[str]
    estimated_switch_cost_ms: float
    sum_token_cost: float
    job_score: float | None
    token_scale: float
    switch_t_scale_ms: float


@dataclass(frozen=True)
class SubtaskTelemetry:
    event: str  # "subtask"
    ts_ms: float
    run_id: str
    job_id: str
    subtask_id: str
    task_type: str
    difficulty: float
    estimated_tokens: int
    assigned_agent: str
    agent_id: str  # "<role>|<model>"
    routing_mode: str
    estimated_performance: float
    estimated_token_cost: float
    estimated_switch_cost_norm: float
    swapped_in_for_job: bool
    actual_latency_ms: float | None
    input_tokens: int | None
    output_tokens: int | None
    judge_score: float | None
    benchmark_name: str | None
    benchmark_score: float | None
    success: bool
    error: str | None = None


@dataclass(frozen=True)
class OrchestrationTelemetry:
    event: str  # "orchestration"
    ts_ms: float
    run_id: str
    job_id: str
    benchmark_name: str | None
    routing_mode: str  # "lp" | "llm"
    routing_source: str  # "lp" | "llm_planner" | "fallback"
    planner_model: str | None
    assignments: list[dict]  # [{subtask_id, role, model, agent_id}]
    active_models: list[str]
    active_role_agents: list[str]


class TelemetryLogger:
    def __init__(self, *, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "telemetry.jsonl"
        # make sure directory exists even if no writes happen yet
        self.path.touch(exist_ok=True)

    def write(self, obj: Any) -> None:
        line = json.dumps(obj, ensure_ascii=False)
        # Defensive: ensure parent exists even if external cleanup happens.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def log_job(self, jt: JobTelemetry) -> None:
        self.write(asdict(jt))

    def log_subtask(self, st: SubtaskTelemetry) -> None:
        self.write(asdict(st))

    def log_orchestration(self, ot: OrchestrationTelemetry) -> None:
        self.write(asdict(ot))


def now_ms() -> float:
    return time.time() * 1000.0


def make_run_dir(base: str = "runs", *, run_id: str) -> Path:
    base_path = Path(base)
    base_path.mkdir(parents=True, exist_ok=True)
    d = base_path / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def default_run_id() -> str:
    # readable + unique-ish without extra deps
    return time.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"

