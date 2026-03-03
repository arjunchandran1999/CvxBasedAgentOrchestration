from __future__ import annotations

import random
from dataclasses import dataclass

from .agents import Agent
from .estimator import QualityEstimator, normalized_switch_cost
from .routing import Assignment, RoutingResult
from .tasks import Subtask
from .router_training.train_router import TrainedRouterModel


@dataclass(frozen=True)
class PrefRouterConfig:
    tau: float = 0.5


def _vram_used(active_models: set[str], agents: list[Agent]) -> float:
    by = {a.name: a for a in agents}
    return float(sum(by[m].vram_gb for m in active_models if m in by))


def route_with_trained_model(
    *,
    subtasks: list[Subtask],
    agents: list[Agent],
    estimator: QualityEstimator,
    router_model: TrainedRouterModel,
    cfg: PrefRouterConfig,
    gpu_vram_gb: float,
    loaded_models: set[str],
    token_scale: float = 1000.0,
    switch_t_scale_ms: float = 1500.0,
) -> RoutingResult:
    """
    Route each subtask using a learned strong-vs-weak scorer.
    """
    estimator.ensure_priors(agents)
    by = {a.name: a for a in agents}
    strong = router_model.strong_model
    weak = router_model.weak_model
    if strong not in by or weak not in by:
        # Cannot route if models not present in agent set.
        return RoutingResult(
            assignments=[],
            active_models=[],
            vram_used_gb=0.0,
            vram_violation=False,
            routing_source="pref_router",
            loaded_models=sorted(loaded_models),
        )

    assignments: list[Assignment] = []
    chosen_models: set[str] = set()
    for s in subtasks:
        p_strong = float(router_model.predict_proba_strong(s.description))
        chosen = strong if p_strong >= float(cfg.tau) else weak
        chosen_models.add(chosen)
        ag = by[chosen]
        quality = estimator.predict(ag, s)
        tok = estimator.estimate_token_cost(ag, s)
        sw = 0.0 if ag.name in loaded_models else normalized_switch_cost(ag, t_scale_ms=switch_t_scale_ms)
        assignments.append(
            Assignment(
                subtask_id=s.id,
                agent=ag.name,
                mode="pref_router",
                estimated_performance=float(quality),
                estimated_token_cost=float(tok),
                estimated_switch_cost=float(sw),
            )
        )

    vram_used = _vram_used(chosen_models, agents)
    vram_violation = vram_used > gpu_vram_gb + 1e-6
    return RoutingResult(
        assignments=assignments,
        active_models=sorted(chosen_models),
        vram_used_gb=float(vram_used),
        vram_violation=bool(vram_violation),
        routing_source="pref_router",
        loaded_models=sorted(loaded_models),
    )


def route_random(
    *,
    subtasks: list[Subtask],
    agents: list[Agent],
    estimator: QualityEstimator,
    gpu_vram_gb: float,
    loaded_models: set[str],
    token_scale: float = 1000.0,
    switch_t_scale_ms: float = 1500.0,
    seed: int = 0,
) -> RoutingResult:
    estimator.ensure_priors(agents)
    rnd = random.Random(seed)
    by = {a.name: a for a in agents}

    # Pick a feasible active set: try loaded models first, else smallest-first.
    candidates = [a for a in agents if a.name in loaded_models] or sorted(agents, key=lambda x: x.vram_gb)
    active: list[Agent] = []
    for a in candidates:
        if sum(m.vram_gb for m in active) + a.vram_gb <= gpu_vram_gb:
            active.append(a)
    if not active:
        active = [min(agents, key=lambda x: x.vram_gb)]

    assignments: list[Assignment] = []
    for s in subtasks:
        ag = rnd.choice(active)
        quality = estimator.predict(ag, s)
        tok = estimator.estimate_token_cost(ag, s)
        sw = 0.0 if ag.name in loaded_models else normalized_switch_cost(ag, t_scale_ms=switch_t_scale_ms)
        assignments.append(
            Assignment(
                subtask_id=s.id,
                agent=ag.name,
                mode="random",
                estimated_performance=float(quality),
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
        routing_source="random",
        loaded_models=sorted(loaded_models),
    )

