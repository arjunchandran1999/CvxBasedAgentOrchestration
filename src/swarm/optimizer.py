from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from .agents import Agent
from .estimator import QualityEstimator, estimate_token_cost, normalized_switch_cost
from .routing import Assignment, RoutingResult
from .tasks import Subtask


@dataclass(frozen=True)
class OptimizerConfig:
    lambda_token: float = 0.5
    lambda_switch: float = 0.2
    token_scale: float = 1000.0
    switch_t_scale_ms: float = 1500.0
    y_active_threshold: float = 0.1
    max_subtasks_per_agent: int | None = None
    solver: str = "SCS"  # "SCS" is widely available for LPs


class SwarmOptimizer:
    def __init__(self, *, config: OptimizerConfig) -> None:
        self.config = config

    def solve(
        self,
        *,
        subtasks: list[Subtask],
        agents: list[Agent],
        estimator: QualityEstimator,
        gpu_vram_gb: float,
        loaded_models: set[str],
    ) -> RoutingResult:
        if not subtasks:
            return RoutingResult(
                assignments=[], active_models=[], vram_used_gb=0.0, routing_source="lp",
                loaded_models=sorted(loaded_models),
            )

        estimator.ensure_priors(agents)

        I = len(agents)
        M = len(subtasks)

        # Matrices: perf and token cost for each (i,m)
        P = np.zeros((I, M), dtype=float)
        Ctok = np.zeros((I, M), dtype=float)
        for i, a in enumerate(agents):
            for m, s in enumerate(subtasks):
                P[i, m] = estimator.predict(a, s)
                Ctok[i, m] = estimate_token_cost(a, s) / float(self.config.token_scale)

        # Switch costs per model (zero if already loaded).
        Csw = np.zeros(I, dtype=float)
        for i, a in enumerate(agents):
            Csw[i] = 0.0 if a.name in loaded_models else normalized_switch_cost(a, t_scale_ms=self.config.switch_t_scale_ms)

        vram = np.array([a.vram_gb for a in agents], dtype=float)

        x = cp.Variable((I, M), nonneg=True)
        y = cp.Variable(I, nonneg=True)

        obj = cp.sum(cp.multiply(x, P - self.config.lambda_token * Ctok)) - self.config.lambda_switch * (Csw @ y)

        constraints = []
        constraints.append(x <= 1)
        constraints.append(y <= 1)

        # Exactly one agent per subtask.
        constraints.append(cp.sum(x, axis=0) == 1)

        # Capacity per agent (vector). Default: each agent can take all subtasks.
        cap = self.config.max_subtasks_per_agent if self.config.max_subtasks_per_agent is not None else M
        capacities = np.full(I, float(cap), dtype=float)
        constraints.append(cp.sum(x, axis=1) <= capacities)

        # Link x and y.
        # For all i,m: x[i,m] <= y[i]
        constraints.append(x <= cp.reshape(y, (I, 1), order="C"))

        # VRAM capacity over active models.
        constraints.append(vram @ y <= float(gpu_vram_gb))

        problem = cp.Problem(cp.Maximize(obj), constraints)

        try:
            problem.solve(solver=self.config.solver, verbose=False)
        except Exception:
            return self._fallback(
                subtasks=subtasks,
                agents=agents,
                estimator=estimator,
                gpu_vram_gb=gpu_vram_gb,
                loaded_models=loaded_models,
            )

        if x.value is None or y.value is None:
            return self._fallback(
                subtasks=subtasks,
                agents=agents,
                estimator=estimator,
                gpu_vram_gb=gpu_vram_gb,
                loaded_models=loaded_models,
            )

        xv = np.asarray(x.value, dtype=float)
        yv = np.asarray(y.value, dtype=float)

        assignments: list[Assignment] = []
        for m, s in enumerate(subtasks):
            i = int(np.argmax(xv[:, m]))
            a = agents[i]
            assignments.append(
                Assignment(
                    subtask_id=s.id,
                    agent=a.name,
                    mode="lp",
                    estimated_performance=float(P[i, m]),
                    estimated_token_cost=float(Ctok[i, m]) * float(self.config.token_scale),
                    estimated_switch_cost=float(Csw[i]),
                )
            )

        active_models = [agents[i].name for i in range(I) if float(yv[i]) > self.config.y_active_threshold]
        # Ensure at least the actually-assigned models are included.
        used_models = sorted({a.agent for a in assignments})
        active_set = set(active_models) | set(used_models)
        active_models = sorted(active_set)

        vram_used = sum(a.vram_gb for a in agents if a.name in active_set)

        # If the relaxed solution includes too many models, enforce VRAM by trimming
        # to the models actually used by assignments (always feasible if assignments feasible).
        if vram_used > gpu_vram_gb + 1e-6:
            active_models = used_models
            vram_used = sum(a.vram_gb for a in agents if a.name in set(active_models))

        vram_violation = vram_used > gpu_vram_gb + 1e-6
        obj_val = float(problem.value) if problem.value is not None else None
        return RoutingResult(
            assignments=assignments,
            active_models=active_models,
            vram_used_gb=float(vram_used),
            vram_violation=bool(vram_violation),
            routing_source="lp",
            loaded_models=sorted(loaded_models),
            lp_objective_value=obj_val,
        )

    def _fallback(
        self,
        *,
        subtasks: list[Subtask],
        agents: list[Agent],
        estimator: QualityEstimator,
        gpu_vram_gb: float,
        loaded_models: set[str],
    ) -> RoutingResult:
        # Simple greedy fallback that tries to stay within VRAM by preferring loaded models.
        estimator.ensure_priors(agents)

        # Candidate model set: loaded models first; otherwise smallest models until VRAM fills.
        loaded = [a for a in agents if a.name in loaded_models]
        if loaded:
            active = loaded
        else:
            active = []
            for a in sorted(agents, key=lambda x: x.vram_gb):
                if sum(m.vram_gb for m in active) + a.vram_gb <= gpu_vram_gb:
                    active.append(a)
            if not active:
                active = [min(agents, key=lambda x: x.vram_gb)]

        active_set = {a.name for a in active}

        assignments: list[Assignment] = []
        for s in subtasks:
            # Choose best utility among active models.
            best = None
            for a in active:
                p = estimator.predict(a, s)
                tok = estimate_token_cost(a, s)
                sw = 0.0 if a.name in loaded_models else normalized_switch_cost(a, t_scale_ms=self.config.switch_t_scale_ms)
                util = p - self.config.lambda_token * (tok / self.config.token_scale) - self.config.lambda_switch * sw
                if best is None or util > best[0]:
                    best = (util, a, p, tok, sw)
            assert best is not None
            _, a, p, tok, sw = best
            assignments.append(
                Assignment(
                    subtask_id=s.id,
                    agent=a.name,
                    mode="fallback",
                    estimated_performance=float(p),
                    estimated_token_cost=float(tok),
                    estimated_switch_cost=float(sw),
                )
            )

        vram_used = sum(a.vram_gb for a in agents if a.name in active_set)
        vram_violation = vram_used > gpu_vram_gb + 1e-6
        return RoutingResult(
            assignments=assignments,
            active_models=sorted(active_set),
            vram_used_gb=float(vram_used),
            vram_violation=bool(vram_violation),
            routing_source="fallback",
            loaded_models=sorted(loaded_models),
        )

