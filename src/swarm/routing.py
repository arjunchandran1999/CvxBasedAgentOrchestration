from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Assignment:
    subtask_id: str
    agent: str
    mode: str  # "lp" | "llm" | "fallback"
    estimated_performance: float
    estimated_token_cost: float
    estimated_switch_cost: float  # normalized for objective (not ms)


@dataclass(frozen=True)
class RoutingResult:
    assignments: list[Assignment]
    active_models: list[str]
    vram_used_gb: float
    vram_violation: bool = False
    routing_source: str = "unknown"  # "lp" | "llm_planner" | "fallback"
    loaded_models: list[str] = field(default_factory=list)  # models at optimizer/planner input (for logging)
    lp_objective_value: float | None = None  # LP optimal objective (when routing_source="lp")

