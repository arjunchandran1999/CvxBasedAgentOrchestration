from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from .agents import Agent
from .estimator import QualityEstimator, normalized_switch_cost
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
    max_subtasks: int = 32
    solver: str = "HIGHS"


class DppSwarmOptimizer:
    """
    DPP-compliant LP optimizer: builds problem once with Parameters, reuses across solves.
    """

    def __init__(self, num_agents: int, max_subtasks: int, *, config: OptimizerConfig) -> None:
        self.num_agents = num_agents
        self.max_subtasks = max_subtasks
        self.config = config

        N, M = num_agents, max_subtasks
        self.x = cp.Variable((N, M), nonneg=True)
        self.y = cp.Variable(N, nonneg=True)

        self.U = cp.Parameter((N, M))
        self.C_sw_penalty = cp.Parameter(N, nonneg=True)
        self.vram = cp.Parameter(N, nonneg=True)
        self.G = cp.Parameter(nonneg=True)
        self.subtask_mask = cp.Parameter(M, nonneg=True)

        self._build_problem()

    def _build_problem(self) -> None:
        x, y = self.x, self.y
        U = self.U
        C_sw_penalty = self.C_sw_penalty
        vram, G = self.vram, self.G
        mask = self.subtask_mask

        N, M = x.shape
        mask_1M = cp.reshape(mask, (1, M), order="C")

        masked_x = cp.multiply(x, mask_1M)
        # DPP: U and C_sw_penalty are parameters, x and y are variables.
        # Use parameter * variable only (avoid parameter * parameter-affine).
        obj = cp.sum(cp.multiply(U, x)) - cp.sum(cp.multiply(C_sw_penalty, y))
        objective = cp.Maximize(obj)

        constraints = [
            cp.sum(masked_x, axis=0) == mask,
            x <= cp.reshape(y, (N, 1), order="C"),
            cp.sum(cp.multiply(vram, y)) <= G,
            x <= 1,
            y <= 1,
        ]
        cap = self.config.max_subtasks_per_agent if self.config.max_subtasks_per_agent is not None else M
        constraints.append(cp.sum(x, axis=1) <= float(cap))

        self.problem = cp.Problem(objective, constraints)
        if not self.problem.is_dcp(dpp=True):
            raise ValueError("Problem is not DPP compliant")

    def solve_step(
        self,
        P_np: np.ndarray,
        C_tok_np: np.ndarray,
        C_sw_np: np.ndarray,
        vram_np: np.ndarray,
        G_value: float,
        lambda_token_value: float,
        lambda_switch_value: float,
        num_subtasks: int,
        *,
        solver: str = "HIGHS",
    ) -> tuple[str | None, np.ndarray | None, np.ndarray | None, float | None]:
        N, M = self.num_agents, self.max_subtasks
        assert P_np.shape[0] == N and P_np.shape[1] <= M
        assert C_tok_np.shape[0] == N and C_tok_np.shape[1] <= M
        assert C_sw_np.shape[0] == N and vram_np.shape[0] == N

        P_full = np.zeros((N, M), dtype=float)
        C_tok_full = np.zeros((N, M), dtype=float)
        P_full[:, : P_np.shape[1]] = P_np
        C_tok_full[:, : C_tok_np.shape[1]] = C_tok_np

        mask = np.zeros(M, dtype=float)
        mask[:num_subtasks] = 1.0
        mask_1M = mask.reshape(1, -1)
        U_np = (P_full - lambda_token_value * C_tok_full) * mask_1M
        C_sw_penalty_np = lambda_switch_value * C_sw_np

        self.U.value = U_np
        self.C_sw_penalty.value = C_sw_penalty_np
        self.vram.value = vram_np
        self.G.value = G_value
        self.subtask_mask.value = mask

        solver_cp = cp.HIGHS if solver.upper() == "HIGHS" else cp.SCS
        try:
            self.problem.solve(solver=solver_cp, verbose=False)
        except Exception:
            try:
                self.problem.solve(solver=cp.SCS, verbose=False)
                solver = "SCS"
            except Exception:
                return None, None, None, None

        status = str(self.problem.status) if hasattr(self.problem, "status") else None
        solve_time = None
        if hasattr(self.problem, "solver_stats") and self.problem.solver_stats is not None:
            t = getattr(self.problem.solver_stats, "solve_time", None)
            if t is not None:
                solve_time = float(t) * 1000.0

        x_val = np.asarray(self.x.value, dtype=float) if self.x.value is not None else None
        y_val = np.asarray(self.y.value, dtype=float) if self.y.value is not None else None
        return status, x_val, y_val, solve_time


class SwarmOptimizer:
    """
    High-level optimizer: builds matrices from subtasks/agents, delegates to DppSwarmOptimizer.
    """

    def __init__(self, *, config: OptimizerConfig) -> None:
        self.config = config
        self._dpp: DppSwarmOptimizer | None = None

    def _get_dpp(self, num_agents: int) -> DppSwarmOptimizer:
        if self._dpp is None or self._dpp.num_agents != num_agents:
            self._dpp = DppSwarmOptimizer(
                num_agents=num_agents,
                max_subtasks=self.config.max_subtasks,
                config=self.config,
            )
        return self._dpp

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
                assignments=[],
                active_models=[],
                vram_used_gb=0.0,
                routing_source="lp",
                loaded_models=sorted(loaded_models),
            )

        estimator.ensure_priors(agents)
        I = len(agents)
        M = len(subtasks)

        P = np.zeros((I, M), dtype=float)
        Ctok = np.zeros((I, M), dtype=float)
        for i, a in enumerate(agents):
            for m, s in enumerate(subtasks):
                P[i, m] = estimator.predict(a, s)
                Ctok[i, m] = estimator.estimate_token_cost(a, s) / float(self.config.token_scale)

        Csw = np.zeros(I, dtype=float)
        for i, a in enumerate(agents):
            Csw[i] = (
                0.0
                if a.name in loaded_models
                else normalized_switch_cost(a, t_scale_ms=self.config.switch_t_scale_ms)
            )

        vram = np.array([a.vram_gb for a in agents], dtype=float)

        dpp = self._get_dpp(I)
        status, x_val, y_val, solve_time_ms = dpp.solve_step(
            P,
            Ctok,
            Csw,
            vram,
            G_value=float(gpu_vram_gb),
            lambda_token_value=self.config.lambda_token,
            lambda_switch_value=self.config.lambda_switch,
            num_subtasks=M,
            solver=self.config.solver,
        )

        if x_val is None or y_val is None:
            return self._fallback(
                subtasks=subtasks,
                agents=agents,
                estimator=estimator,
                gpu_vram_gb=gpu_vram_gb,
                loaded_models=loaded_models,
            )

        xv = x_val[:, :M]
        yv = y_val

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
        used_models = sorted({a.agent for a in assignments})
        active_set = set(active_models) | set(used_models)
        active_models = sorted(active_set)

        vram_used = sum(a.vram_gb for a in agents if a.name in active_set)
        if vram_used > gpu_vram_gb + 1e-6:
            active_models = used_models
            vram_used = sum(a.vram_gb for a in agents if a.name in set(active_models))

        vram_violation = vram_used > gpu_vram_gb + 1e-6
        obj_val = float(self._dpp.problem.value) if self._dpp and self._dpp.problem.value is not None else None

        solver_name = self.config.solver
        if hasattr(dpp.problem, "solver_stats") and dpp.problem.solver_stats is not None:
            sn = getattr(dpp.problem.solver_stats, "solver_name", None)
            if sn:
                solver_name = str(sn)

        return RoutingResult(
            assignments=assignments,
            active_models=active_models,
            vram_used_gb=float(vram_used),
            vram_violation=bool(vram_violation),
            routing_source="lp",
            loaded_models=sorted(loaded_models),
            lp_objective_value=obj_val,
            lp_solver_name=solver_name,
            lp_solve_status=status,
            lp_solve_time_ms=solve_time_ms,
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
        estimator.ensure_priors(agents)
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
            best = None
            for a in active:
                p = estimator.predict(a, s)
                tok = estimator.estimate_token_cost(a, s)
                sw = (
                    0.0
                    if a.name in loaded_models
                    else normalized_switch_cost(a, t_scale_ms=self.config.switch_t_scale_ms)
                )
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
