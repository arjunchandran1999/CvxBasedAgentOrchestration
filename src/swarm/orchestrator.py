from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .agents import Agent, default_agents
from .estimator import QualityEstimator, estimate_token_cost, normalized_switch_cost
from .executor import SubtaskResult, execute
from .evaluator import Evaluation, evaluate
from .ollama_client import OllamaClient
from .optimizer import OptimizerConfig, SwarmOptimizer
from .planner_router import llm_assign
from .routing import Assignment, RoutingResult
from .tasks import Subtask, decompose
from .task_graph import TaskDAG, get_horizon_nodes, get_layers, node_to_subtask, topological_sort
from typing import Literal

from .benchmarks.registry import BenchmarkExample, ensure_default_benchmarks_loaded, get as get_benchmark

from .telemetry import JobTelemetry, OrchestrationTelemetry, SubtaskPlanTelemetry, SubtaskTelemetry, TelemetryLogger, now_ms
from .gpu import GpuSpec, detect_gpu, get_effective_vram_gb


def _compute_estimates_and_oracle(
    *,
    subtasks: list,
    agents: list[Agent],
    estimator: QualityEstimator,
    loaded_models: set[str],
    lambda_token: float,
    lambda_switch: float,
    token_scale: float,
    switch_t_scale_ms: float,
    assignments: list[Assignment],
    lp_objective_value: float | None,
) -> tuple[list[dict], dict]:
    """Compute per (model, task) estimates and oracle (best-per-task, LP objective)."""
    by_name = {a.name: a for a in agents}
    model_task_estimates: list[dict] = []
    best_by_quality: dict[str, str] = {}
    best_by_utility: dict[str, str] = {}

    for s in subtasks:
        best_q = None
        best_q_model = None
        best_u = None
        best_u_model = None
        for a in agents:
            quality = estimator.predict(a, s)
            tok_raw = estimate_token_cost(a, s)
            tok_norm = tok_raw / token_scale
            sw = 0.0 if a.name in loaded_models else normalized_switch_cost(a, t_scale_ms=switch_t_scale_ms)
            utility_cell = quality - lambda_token * tok_norm  # cell utility before switch
            model_task_estimates.append({
                "subtask_id": s.id,
                "model": a.name,
                "quality": round(quality, 4),
                "token_cost": round(tok_raw, 2),
                "switch_cost": round(sw, 4),
                "utility_cell": round(utility_cell, 4),
                "utility": round(utility_cell - lambda_switch * sw, 4),  # includes switch if unloaded
            })
            if best_q is None or quality > best_q:
                best_q = quality
                best_q_model = a.name
            util = utility_cell - lambda_switch * sw
            if best_u is None or util > best_u:
                best_u = util
                best_u_model = a.name
        best_by_quality[s.id] = best_q_model
        best_by_utility[s.id] = best_u_model

    # Chosen assignment's total utility (LP objective: sum(P - λ*Ctok) - λ_switch * sum(Csw for used unloaded models))
    chosen_models = {a.subtask_id: a.agent for a in assignments}
    used_models = set(chosen_models.values())
    chosen_utility = 0.0
    for e in model_task_estimates:
        if e["subtask_id"] in chosen_models and e["model"] == chosen_models[e["subtask_id"]]:
            chosen_utility += e["utility_cell"]
    switch_penalty = 0.0
    for m in used_models:
        if m not in loaded_models and m in by_name:
            switch_penalty += lambda_switch * normalized_switch_cost(by_name[m], t_scale_ms=switch_t_scale_ms)
    chosen_utility -= switch_penalty

    oracle: dict = {
        "best_per_task_by_quality": best_by_quality,
        "best_per_task_by_utility": best_by_utility,
        "chosen_utility": round(chosen_utility, 4),
    }
    if lp_objective_value is not None:
        oracle["lp_objective_value"] = round(lp_objective_value, 4)
    return model_task_estimates, oracle


@dataclass(frozen=True)
class JobSummary:
    run_id: str
    job_id: str
    routing_mode: str
    query: str
    final_answer: str | None
    benchmark_name: str | None
    job_score: float | None
    active_models: list[str]
    models_swapped_in: list[str]
    estimated_switch_cost_ms: float
    vram_used_gb: float
    vram_violation: bool
    total_latency_ms: float | None
    total_prompt_tokens: int | None
    total_completion_tokens: int | None
    avg_judge_score: float | None
    success_rate: float
    plan: list[dict]  # subtask plan fed to router: [{id, task_type, description, estimated_tokens, difficulty}]
    task_to_model: dict[str, str]  # subtask_id -> chosen model
    loaded_models: list[str]  # models at router input (from SwarmOptimizer / planner)
    model_task_estimates: list[dict]  # [{subtask_id, model, quality, token_cost, switch_cost, utility}]
    oracle: dict  # {lp_objective_value?, best_per_task_by_quality, best_per_task_by_utility, chosen_utility}


class SwarmOrchestrator:
    def __init__(
        self,
        *,
        run_id: str,
        run_dir: Path,
        routing_mode: Literal["lp", "llm"],
        lambda_token: float,
        lambda_switch: float,
        gpu_vram_gb: float,
        token_scale: float = 1000.0,
        switch_t_scale_ms: float = 1500.0,
        dry_run: bool = False,
        judge: bool = False,
        ollama_base_url: str = "http://localhost:11434",
        agents: list[Agent] | None = None,
        decomposer_model: str = "llama3.1:8b",
        planner_model: str = "llama3.1:8b",
        judge_model: str = "llama3.1:8b",
        planner_timeout_s: float = 120.0,
        horizon_depth: int = 1,
    ) -> None:
        self.run_id = run_id
        self.run_dir = run_dir
        self.routing_mode: Literal["lp", "llm"] = routing_mode
        self.lambda_token = float(lambda_token)
        self.lambda_switch = float(lambda_switch)
        self.gpu_spec: GpuSpec | None = detect_gpu()
        # If caller passed a positive value, treat it as explicit constraint.
        # Otherwise (0 or negative), fall back to detected VRAM.
        if float(gpu_vram_gb) > 0:
            self.gpu_vram_gb = float(gpu_vram_gb)
        else:
            self.gpu_vram_gb = get_effective_vram_gb()
        self.token_scale = float(token_scale)
        self.switch_t_scale_ms = float(switch_t_scale_ms)
        self.dry_run = dry_run
        self.judge = judge

        self.agents = agents or default_agents()
        self.estimator = QualityEstimator()
        self.estimator.ensure_priors(self.agents)

        self.ollama = OllamaClient(base_url=ollama_base_url, dry_run=dry_run)
        self.telemetry = TelemetryLogger(run_dir=run_dir)
        self.decomposer_model = str(decomposer_model)
        self.planner_model = str(planner_model)
        self.judge_model = str(judge_model)
        self.planner_timeout_s = float(planner_timeout_s)
        self.horizon_depth = max(0, int(horizon_depth))

        self.optimizer = SwarmOptimizer(
            config=OptimizerConfig(
                lambda_token=self.lambda_token,
                lambda_switch=self.lambda_switch,
                token_scale=self.token_scale,
                switch_t_scale_ms=self.switch_t_scale_ms,
            )
        )

        self.loaded_models: set[str] = set()

    async def run_job(
        self,
        *,
        query: str,
        job_id: str,
        benchmark_name: str | None = None,
        benchmark_reference: dict | None = None,
        subtasks_override: list[Subtask] | None = None,
        job_score: float | None = None,
    ) -> JobSummary:
        loaded_before = set(self.loaded_models)
        # Minimal early event so we can diagnose hangs in planning/execution.
        # (Uses placeholder fields that will be overwritten by later 'job'/'job_costs' events.)
        self.telemetry.log_job(
            JobTelemetry(
                event="job_start",
                ts_ms=now_ms(),
                run_id=self.run_id,
                job_id=job_id,
                query=query,
                gpu_name=self.gpu_spec.name if self.gpu_spec else None,
                gpu_uuid=self.gpu_spec.uuid if self.gpu_spec else None,
                benchmark_name=benchmark_name,
                routing_mode=self.routing_mode,
                lambda_token=self.lambda_token,
                lambda_switch=self.lambda_switch,
                gpu_vram_gb=self.gpu_vram_gb,
                vram_used_gb=0.0,
                vram_violation=False,
                loaded_models_before=sorted(loaded_before),
                active_models=[],
                active_roles=[],
                active_role_agents=[],
                n_distinct_models=0,
                n_role_agents=0,
                models_swapped_in=[],
                estimated_switch_cost_ms=0.0,
                sum_token_cost=0.0,
                job_score=None,
                token_scale=self.token_scale,
                switch_t_scale_ms=self.switch_t_scale_ms,
            )
        )

        subtasks: list[Subtask]
        if subtasks_override is not None:
            subtasks = subtasks_override
            plan_source = "benchmark"
        else:
            subtasks = await decompose(query, self.ollama, planner_model=self.decomposer_model)
            plan_source = "decomposer"

        # Log the subtask plan fed to LP or LLM planner before routing.
        self.telemetry.log_subtask_plan(
            SubtaskPlanTelemetry(
                event="subtask_plan",
                ts_ms=now_ms(),
                run_id=self.run_id,
                job_id=job_id,
                benchmark_name=benchmark_name,
                routing_mode=self.routing_mode,
                plan_source=plan_source,
                subtasks=[
                    {
                        "id": s.id,
                        "task_type": s.task_type.value,
                        "description": s.description,
                        "estimated_tokens": s.estimated_tokens,
                        "difficulty": float(s.difficulty),
                    }
                    for s in subtasks
                ],
            )
        )

        if self.routing_mode == "lp":
            routing: RoutingResult = self.optimizer.solve(
                subtasks=subtasks,
                agents=self.agents,
                estimator=self.estimator,
                gpu_vram_gb=self.gpu_vram_gb,
                loaded_models=loaded_before,
            )
        else:
            routing = await llm_assign(
                subtasks,
                self.agents,
                self.estimator,
                ollama=self.ollama,
                planner_model=self.planner_model,
                gpu_vram_gb=self.gpu_vram_gb,
                loaded_models=loaded_before,
                lambda_token=self.lambda_token,
                lambda_switch=self.lambda_switch,
                token_scale=self.token_scale,
                switch_t_scale_ms=self.switch_t_scale_ms,
                planner_timeout_s=self.planner_timeout_s,
            )

        active_models = set(routing.active_models)
        # Role-aware agent identities: (task_type, model) pairs.
        assignment_by_subtask = {a.subtask_id: a for a in routing.assignments}
        active_roles = sorted({s.task_type.value for s in subtasks})
        active_role_agents = sorted(
            {
                f"{s.task_type.value}|{assignment_by_subtask[s.id].agent}"
                for s in subtasks
                if s.id in assignment_by_subtask
            }
        )
        models_swapped_in = sorted(active_models - loaded_before)
        by_name = {a.name: a for a in self.agents}
        estimated_switch_cost_ms = float(sum(by_name[m].load_time_ms for m in models_swapped_in if m in by_name))

        orchestration_assignments: list[dict] = []
        for s in subtasks:
            a = assignment_by_subtask.get(s.id)
            if a is None:
                continue
            orchestration_assignments.append(
                {
                    "subtask_id": s.id,
                    "role": s.task_type.value,
                    "model": a.agent,
                    "agent_id": f"{s.task_type.value}|{a.agent}",
                }
            )

        # Compute estimates and oracle (before estimator update) for logging
        lp_obj = getattr(routing, "lp_objective_value", None)
        model_task_estimates, oracle = _compute_estimates_and_oracle(
            subtasks=subtasks,
            agents=self.agents,
            estimator=self.estimator,
            loaded_models=loaded_before,
            lambda_token=self.lambda_token,
            lambda_switch=self.lambda_switch,
            token_scale=self.token_scale,
            switch_t_scale_ms=self.switch_t_scale_ms,
            assignments=routing.assignments,
            lp_objective_value=lp_obj,
        )

        # Explicit orchestration output log (tagged by routing source).
        self.telemetry.log_orchestration(
            OrchestrationTelemetry(
                event="orchestration",
                ts_ms=now_ms(),
                run_id=self.run_id,
                job_id=job_id,
                benchmark_name=benchmark_name,
                routing_mode=self.routing_mode,
                routing_source=str(getattr(routing, "routing_source", "unknown")),
                planner_model=(self.planner_model if self.routing_mode == "llm" else None),
                assignments=orchestration_assignments,
                active_models=sorted(active_models),
                active_role_agents=active_role_agents,
            )
        )

        # Execute all subtasks.
        results: list[SubtaskResult] = await execute(
            query=query,
            subtasks=subtasks,
            assignments=routing.assignments,
            ollama=self.ollama,
            loaded_models_before=loaded_before,
            max_concurrency=4,
        )

        # Evaluate outputs.
        evals: list[Evaluation] = await evaluate(
            subtasks=subtasks,
            results=results,
            judge=self.judge,
            ollama=self.ollama,
            judge_model=self.judge_model,
        )
        eval_by_id = {e.subtask_id: e for e in evals}

        # Benchmark scoring (job_score + per-subtask scores) if provided.
        computed_job_score: float | None = None
        computed_subtask_scores: dict[str, float] = {}
        if benchmark_name is not None:
            try:
                ensure_default_benchmarks_loaded()
                bench = get_benchmark(benchmark_name)
                ex = BenchmarkExample(
                    benchmark=benchmark_name,
                    example_id=str(benchmark_reference.get("job_id") if isinstance(benchmark_reference, dict) and "job_id" in benchmark_reference else job_id),
                    query=query,
                    reference=benchmark_reference or {},
                )
                tmp_artifact = {
                    "results": [asdict(r) for r in results],
                    "subtasks": [asdict(s) for s in subtasks],
                    "benchmark_reference": benchmark_reference or {},
                }
                if hasattr(bench, "score_artifact"):
                    scored = bench.score_artifact(example=ex, artifact=tmp_artifact)  # type: ignore[attr-defined]
                    computed_job_score = float(scored.get("score", 0.0))
                    # Optional subtask scores for workflowbench
                    sub_scores = scored.get("subtask_scores")
                    if isinstance(sub_scores, list):
                        # Map in order of benchmark reference subtasks if present
                        refs = (benchmark_reference or {}).get("subtasks") if isinstance(benchmark_reference, dict) else None
                        if isinstance(refs, list) and len(refs) == len(sub_scores):
                            for ref_s, sc in zip(refs, sub_scores):
                                sid = str(ref_s.get("id"))
                                computed_subtask_scores[sid] = float(sc)
                elif hasattr(bench, "score"):
                    computed_job_score = None
            except Exception:
                computed_job_score = None

        # Synthesize a single final answer (benchmark-friendly).
        final_answer: str | None = None
        # For benchmark runs, skip synthesis to avoid adding extra (untracked) model calls
        # that would confound routing/switching efficiency comparisons.
        if benchmark_name is None:
            try:
                synth_system = "You are the final responder. Combine subtask outputs into a single final answer for the user."
                result_by_id = {r.subtask_id: r for r in results}
                synth_user = {
                    "query": query,
                    "subtasks": [
                        {
                            "id": s.id,
                            "type": s.task_type.value,
                            "description": s.description,
                            "agent": (result_by_id.get(s.id).agent if result_by_id.get(s.id) else None),
                            "output": (result_by_id.get(s.id).output if result_by_id.get(s.id) else ""),
                            "success": (result_by_id.get(s.id).success if result_by_id.get(s.id) else False),
                        }
                        for s in subtasks
                    ],
                    "rules": [
                        "Be concise but complete.",
                        "If the task expects a single value (multiple-choice, numeric), output it clearly.",
                        "Do not mention internal routing/telemetry.",
                    ],
                }
                final_answer = await self.ollama.chat(
                    model="llama3.1:8b",
                    system=synth_system,
                    user=json.dumps(synth_user, ensure_ascii=False),
                    temperature=0.2,
                )
            except Exception:
                final_answer = None

        # Update estimator.
        subtask_by_id = {s.id: s for s in subtasks}
        agent_by_name = {a.name: a for a in self.agents}
        for a in routing.assignments:
            s = subtask_by_id.get(a.subtask_id)
            if s is None:
                continue
            ev = eval_by_id.get(a.subtask_id)
            if ev is None:
                continue
            observed = ev.judge_score if ev.judge_score is not None else (1.0 if ev.success else 0.0)
            ag = agent_by_name.get(a.agent)
            if ag is None:
                continue
            self.estimator.update(agent=ag, subtask_type=s.task_type, observed_score=float(observed))

        # Telemetry: per-job
        self.telemetry.log_job(
            JobTelemetry(
                event="job",
                ts_ms=now_ms(),
                run_id=self.run_id,
                job_id=job_id,
                query=query,
                gpu_name=self.gpu_spec.name if self.gpu_spec else None,
                gpu_uuid=self.gpu_spec.uuid if self.gpu_spec else None,
                benchmark_name=benchmark_name,
                routing_mode=self.routing_mode,
                lambda_token=self.lambda_token,
                lambda_switch=self.lambda_switch,
                gpu_vram_gb=self.gpu_vram_gb,
                vram_used_gb=float(routing.vram_used_gb),
                vram_violation=bool(routing.vram_violation),
                loaded_models_before=sorted(loaded_before),
                active_models=sorted(active_models),
                active_roles=active_roles,
                active_role_agents=active_role_agents,
                n_distinct_models=len(active_models),
                n_role_agents=len(active_role_agents),
                models_swapped_in=models_swapped_in,
                estimated_switch_cost_ms=estimated_switch_cost_ms,
                sum_token_cost=0.0,  # filled after subtask loop
                job_score=job_score if job_score is not None else computed_job_score,
                token_scale=self.token_scale,
                switch_t_scale_ms=self.switch_t_scale_ms,
            )
        )

        # Telemetry: per-subtask
        result_by_subtask = {r.subtask_id: r for r in results}

        sum_token_cost = 0.0
        for s in subtasks:
            a = assignment_by_subtask.get(s.id)
            r = result_by_subtask.get(s.id)
            ev = eval_by_id.get(s.id)
            if a is None or r is None:
                continue

            token_used = None
            if isinstance(r.prompt_tokens, int) and isinstance(r.completion_tokens, int):
                token_used = r.prompt_tokens + r.completion_tokens
            else:
                token_used = int(s.estimated_tokens)
            agent_cost = agent_by_name.get(a.agent).cost_per_token if agent_by_name.get(a.agent) else 1.0
            subtask_token_cost = float(agent_cost) * float(token_used)
            sum_token_cost += subtask_token_cost

            self.telemetry.log_subtask(
                SubtaskTelemetry(
                    event="subtask",
                    ts_ms=now_ms(),
                    run_id=self.run_id,
                    job_id=job_id,
                    subtask_id=s.id,
                    task_type=s.task_type.value,
                    difficulty=float(s.difficulty),
                    estimated_tokens=int(s.estimated_tokens),
                    assigned_agent=a.agent,
                    agent_id=f"{s.task_type.value}|{a.agent}",
                    routing_mode=a.mode,
                    estimated_performance=float(a.estimated_performance),
                    estimated_token_cost=float(a.estimated_token_cost),
                    estimated_switch_cost_norm=float(a.estimated_switch_cost),
                    swapped_in_for_job=bool(r.swapped_in_for_job),
                    actual_latency_ms=r.latency_ms,
                    input_tokens=r.prompt_tokens,
                    output_tokens=r.completion_tokens,
                    judge_score=ev.judge_score if ev else None,
                    benchmark_name=benchmark_name,
                    benchmark_score=(computed_subtask_scores.get(s.id) if computed_subtask_scores else None),
                    success=bool(r.success),
                    error=r.error,
                    output=r.output,
                )
            )

        # Patch the job telemetry line with sum_token_cost by appending a follow-up event.
        self.telemetry.log_job(
            JobTelemetry(
                event="job_costs",
                ts_ms=now_ms(),
                run_id=self.run_id,
                job_id=job_id,
                query=query,
                gpu_name=self.gpu_spec.name if self.gpu_spec else None,
                gpu_uuid=self.gpu_spec.uuid if self.gpu_spec else None,
                benchmark_name=benchmark_name,
                routing_mode=self.routing_mode,
                lambda_token=self.lambda_token,
                lambda_switch=self.lambda_switch,
                gpu_vram_gb=self.gpu_vram_gb,
                vram_used_gb=float(routing.vram_used_gb),
                vram_violation=bool(routing.vram_violation),
                loaded_models_before=sorted(loaded_before),
                active_models=sorted(active_models),
                active_roles=active_roles,
                active_role_agents=active_role_agents,
                n_distinct_models=len(active_models),
                n_role_agents=len(active_role_agents),
                models_swapped_in=models_swapped_in,
                estimated_switch_cost_ms=estimated_switch_cost_ms,
                sum_token_cost=float(sum_token_cost),
                job_score=job_score if job_score is not None else computed_job_score,
                token_scale=self.token_scale,
                switch_t_scale_ms=self.switch_t_scale_ms,
            )
        )

        # Aggregate summary
        total_latency = None
        latencies = [r.latency_ms for r in results if isinstance(r.latency_ms, (int, float))]
        if latencies:
            total_latency = float(sum(latencies))

        prompt_tokens = [r.prompt_tokens for r in results if isinstance(r.prompt_tokens, int)]
        completion_tokens = [r.completion_tokens for r in results if isinstance(r.completion_tokens, int)]
        total_prompt = int(sum(prompt_tokens)) if prompt_tokens else None
        total_completion = int(sum(completion_tokens)) if completion_tokens else None

        judge_scores = [e.judge_score for e in evals if isinstance(e.judge_score, (int, float))]
        avg_judge = float(sum(judge_scores) / len(judge_scores)) if judge_scores else None

        successes = [1.0 if r.success else 0.0 for r in results]
        success_rate = float(sum(successes) / len(successes)) if successes else 0.0

        plan = [
            {
                "id": s.id,
                "task_type": s.task_type.value,
                "description": s.description,
                "estimated_tokens": s.estimated_tokens,
                "difficulty": float(s.difficulty),
            }
            for s in subtasks
        ]
        task_to_model = {a.subtask_id: a.agent for a in routing.assignments}
        loaded_models_list = list(getattr(routing, "loaded_models", None) or [])

        summary = JobSummary(
            run_id=self.run_id,
            job_id=job_id,
            routing_mode=self.routing_mode,
            query=query,
            final_answer=final_answer,
            benchmark_name=benchmark_name,
            job_score=job_score if job_score is not None else computed_job_score,
            active_models=sorted(active_models),
            models_swapped_in=models_swapped_in,
            estimated_switch_cost_ms=estimated_switch_cost_ms,
            vram_used_gb=float(routing.vram_used_gb),
            vram_violation=bool(routing.vram_violation),
            total_latency_ms=total_latency,
            total_prompt_tokens=total_prompt,
            total_completion_tokens=total_completion,
            avg_judge_score=avg_judge,
            success_rate=success_rate,
            plan=plan,
            task_to_model=task_to_model,
            loaded_models=loaded_models_list,
            model_task_estimates=model_task_estimates,
            oracle=oracle,
        )

        # Write job artifact for benchmark scoring.
        artifacts_dir = self.run_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        (artifacts_dir / f"{job_id}.json").write_text(
            json.dumps(
                {
                    "run_id": self.run_id,
                    "job_id": job_id,
                    "routing_mode": self.routing_mode,
                    "routing_source": str(getattr(routing, "routing_source", "unknown")),
                    "query": query,
                    "benchmark_name": benchmark_name,
                    "benchmark_reference": benchmark_reference,
                    "subtasks": [asdict(s) for s in subtasks],
                    "assignments": [asdict(a) for a in routing.assignments],
                    "orchestration": {
                        "routing_mode": self.routing_mode,
                        "routing_source": str(getattr(routing, "routing_source", "unknown")),
                        "planner_model": (self.planner_model if self.routing_mode == "llm" else None),
                        "assignments": orchestration_assignments,
                        "active_models": sorted(active_models),
                        "active_role_agents": active_role_agents,
                    },
                    "results": [asdict(r) for r in results],
                    "evaluations": [asdict(e) for e in evals],
                    "final_answer": final_answer,
                    "active_models": sorted(active_models),
                    "active_roles": active_roles,
                    "active_role_agents": active_role_agents,
                    "loaded_models_before": sorted(loaded_before),
                    "loaded_models_at_router": loaded_models_list,
                    "model_task_estimates": model_task_estimates,
                    "oracle": oracle,
                    "vram_used_gb": float(routing.vram_used_gb),
                    "vram_violation": bool(routing.vram_violation),
                    "job_score": job_score if job_score is not None else computed_job_score,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        # Per-job summary (batch-friendly) + latest summary for convenience.
        (self.run_dir / f"summary_{job_id}.json").write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
        (self.run_dir / "summary.json").write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")

        # Update loaded models for next job.
        self.loaded_models = set(active_models)

        return summary

    async def run_job_dag(
        self,
        *,
        task_dag: TaskDAG,
        query: str,
        job_id: str | None = None,
        benchmark_name: str | None = None,
        benchmark_reference: dict | None = None,
        horizon_depth: int | None = None,
    ) -> JobSummary:
        """Execute a job as a DAG with MPC-style receding-horizon planning.

        When horizon_depth=0: route and execute layer-by-layer (pure DAG).
        When horizon_depth>=1: at each step, optimize over ready nodes + successors up to
        horizon_depth steps ahead, then execute only the currently ready subset (MPC).
        """
        h_depth = horizon_depth if horizon_depth is not None else self.horizon_depth
        job_id = job_id or task_dag.job_id
        loaded_before = set(self.loaded_models)

        self.telemetry.log_job(
            JobTelemetry(
                event="job_start",
                ts_ms=now_ms(),
                run_id=self.run_id,
                job_id=job_id,
                query=query,
                gpu_name=self.gpu_spec.name if self.gpu_spec else None,
                gpu_uuid=self.gpu_spec.uuid if self.gpu_spec else None,
                benchmark_name=benchmark_name,
                routing_mode=self.routing_mode,
                lambda_token=self.lambda_token,
                lambda_switch=self.lambda_switch,
                gpu_vram_gb=self.gpu_vram_gb,
                vram_used_gb=0.0,
                vram_violation=False,
                loaded_models_before=sorted(loaded_before),
                active_models=[],
                active_roles=[],
                active_role_agents=[],
                n_distinct_models=0,
                n_role_agents=0,
                models_swapped_in=[],
                estimated_switch_cost_ms=0.0,
                sum_token_cost=0.0,
                job_score=None,
                token_scale=self.token_scale,
                switch_t_scale_ms=self.switch_t_scale_ms,
            )
        )

        topo = topological_sort(task_dag.nodes)
        layers = get_layers(task_dag.nodes, topo)
        all_node_ids = set(topo)
        edges = [(p, c) for n in task_dag.nodes.values() for p in n.depends_on for c in [n.id] if p in task_dag.nodes]
        self.telemetry.log_dag_structure(
            {
                "ts_ms": now_ms(),
                "run_id": self.run_id,
                "job_id": job_id,
                "benchmark_name": benchmark_name,
                "mpc": h_depth > 0,
                "horizon_depth": h_depth,
                "nodes": [{"id": n.id, "task_type": n.task_type.value, "difficulty": n.difficulty} for n in task_dag.nodes.values()],
                "edges": edges,
            }
        )

        all_subtasks = [node_to_subtask(n) for n in task_dag.nodes.values()]
        self.telemetry.log_subtask_plan(
            SubtaskPlanTelemetry(
                event="subtask_plan",
                ts_ms=now_ms(),
                run_id=self.run_id,
                job_id=job_id,
                benchmark_name=benchmark_name,
                routing_mode=self.routing_mode,
                plan_source="benchmark_dag",
                subtasks=[{"id": s.id, "task_type": s.task_type.value, "description": s.description, "estimated_tokens": s.estimated_tokens, "difficulty": float(s.difficulty)} for s in all_subtasks],
            )
        )

        completed: set[str] = set()
        results_all: list[SubtaskResult] = []
        evals_all: list[Evaluation] = []
        plan_step_index = 0
        total_estimated_switch_ms = 0.0
        by_name = {a.name: a for a in self.agents}
        all_assignments: list[Assignment] = []
        model_task_estimates: list[dict] = []
        oracle: dict = {}
        peak_vram_gb = 0.0  # Peak concurrent VRAM (max over plan steps), not sum of distinct models

        while completed < all_node_ids:
            ready_now = [
                nid for nid in topo
                if nid not in completed and set(task_dag.nodes[nid].depends_on).issubset(completed)
            ]
            if not ready_now:
                break
            horizon_nodes = (
                get_horizon_nodes(completed, task_dag, h_depth)
                if h_depth > 0
                else ready_now
            )
            horizon_subtasks = [node_to_subtask(task_dag.nodes[nid]) for nid in horizon_nodes]
            ready_subtasks = [node_to_subtask(task_dag.nodes[nid]) for nid in ready_now]

            if self.routing_mode == "lp":
                routing = self.optimizer.solve(
                    subtasks=horizon_subtasks,
                    agents=self.agents,
                    estimator=self.estimator,
                    gpu_vram_gb=self.gpu_vram_gb,
                    loaded_models=set(self.loaded_models),
                )
            else:
                routing = await llm_assign(
                    horizon_subtasks,
                    self.agents,
                    self.estimator,
                    ollama=self.ollama,
                    planner_model=self.planner_model,
                    gpu_vram_gb=self.gpu_vram_gb,
                    loaded_models=set(self.loaded_models),
                    lambda_token=self.lambda_token,
                    lambda_switch=self.lambda_switch,
                    token_scale=self.token_scale,
                    switch_t_scale_ms=self.switch_t_scale_ms,
                    planner_timeout_s=self.planner_timeout_s,
                )

            ready_assignments = [a for a in routing.assignments if a.subtask_id in ready_now]

            # Peak VRAM: max concurrent at any step (models used in this step may run in parallel)
            step_models = {a.agent for a in ready_assignments}
            step_vram = sum(by_name[m].vram_gb for m in step_models if m in by_name)
            peak_vram_gb = max(peak_vram_gb, step_vram)

            active_models = set(routing.active_models)
            models_swapped = sorted(active_models - set(self.loaded_models))
            total_estimated_switch_ms += sum(by_name[m].load_time_ms for m in models_swapped if m in by_name)

            plan_diag: dict = {
                "ts_ms": now_ms(),
                "run_id": self.run_id,
                "job_id": job_id,
                "plan_step_index": plan_step_index,
                "routing_mode": self.routing_mode,
                "mpc": h_depth > 0,
                "horizon_depth": h_depth,
                "horizon_nodes": horizon_nodes,
                "ready_now_nodes": ready_now,
                "active_models_before": sorted(self.loaded_models),
                "gpu_vram_gb": self.gpu_vram_gb,
            }
            if hasattr(routing, "lp_solver_name") and routing.lp_solver_name:
                plan_diag["lp_solver_name"] = routing.lp_solver_name
            if hasattr(routing, "lp_solve_status") and routing.lp_solve_status:
                plan_diag["lp_solve_status"] = routing.lp_solve_status
            if hasattr(routing, "lp_solve_time_ms") and routing.lp_solve_time_ms is not None:
                plan_diag["lp_solve_time_ms"] = routing.lp_solve_time_ms
            if getattr(routing, "lp_objective_value", None) is not None:
                plan_diag["lp_objective_value"] = routing.lp_objective_value
            plan_diag["assignments"] = [
                {"subtask_id": a.subtask_id, "assigned_agent": a.agent, "estimated_performance": a.estimated_performance, "estimated_token_cost": a.estimated_token_cost, "estimated_switch_cost_share": a.estimated_switch_cost}
                for a in routing.assignments
            ]
            plan_diag["executed_assignments"] = [
                {"subtask_id": a.subtask_id, "assigned_agent": a.agent}
                for a in ready_assignments
            ]
            plan_diag["step_vram_gb"] = round(step_vram, 3)
            self.telemetry.log_plan_step(plan_diag)

            self.telemetry.log_orchestration(
                OrchestrationTelemetry(
                    event="orchestration",
                    ts_ms=now_ms(),
                    run_id=self.run_id,
                    job_id=job_id,
                    benchmark_name=benchmark_name,
                    routing_mode=self.routing_mode,
                    routing_source=str(getattr(routing, "routing_source", "unknown")),
                    planner_model=self.planner_model if self.routing_mode == "llm" else None,
                    assignments=[{"subtask_id": a.subtask_id, "role": ready_subtasks[i].task_type.value if i < len(ready_subtasks) else "", "model": a.agent, "agent_id": f"{ready_subtasks[i].task_type.value if i < len(ready_subtasks) else ''}|{a.agent}"} for i, a in enumerate(ready_assignments)],
                    active_models=sorted(active_models),
                    active_role_agents=sorted({f"{s.task_type.value}|{a.agent}" for s, a in zip(ready_subtasks, ready_assignments)}),
                )
            )

            results = await execute(
                query=query,
                subtasks=ready_subtasks,
                assignments=ready_assignments,
                ollama=self.ollama,
                loaded_models_before=set(self.loaded_models),
                max_concurrency=4,
            )
            results_all.extend(results)
            all_assignments.extend(ready_assignments)

            evals = await evaluate(subtasks=ready_subtasks, results=results, judge=self.judge, ollama=self.ollama, judge_model=self.judge_model)
            evals_all.extend(evals)
            eval_by_id = {e.subtask_id: e for e in evals}
            subtask_by_id = {s.id: s for s in ready_subtasks}
            agent_by_name = {a.name: a for a in self.agents}
            for a in ready_assignments:
                s = subtask_by_id.get(a.subtask_id)
                if s is None:
                    continue
                ev = eval_by_id.get(a.subtask_id)
                if ev is None:
                    continue
                observed = ev.judge_score if ev.judge_score is not None else (1.0 if ev.success else 0.0)
                ag = agent_by_name.get(a.agent)
                if ag is not None:
                    self.estimator.update(agent=ag, subtask_type=s.task_type, observed_score=float(observed))

            self.loaded_models = active_models

            for a in ready_assignments:
                r = next((x for x in results if x.subtask_id == a.subtask_id), None)
                ev = eval_by_id.get(a.subtask_id)
                s = subtask_by_id.get(a.subtask_id)
                if r is None or s is None:
                    continue
                completed.add(a.subtask_id)
                node = task_dag.nodes.get(a.subtask_id)
                parents = list(node.depends_on) if node else []
                children = [cid for cid, n in task_dag.nodes.items() if a.subtask_id in n.depends_on]
                token_used = (r.prompt_tokens + r.completion_tokens) if isinstance(r.prompt_tokens, int) and isinstance(r.completion_tokens, int) else s.estimated_tokens
                agent_cost = agent_by_name.get(a.agent).cost_per_token if agent_by_name.get(a.agent) else 1.0
                subtask_token_cost = float(agent_cost) * float(token_used)

                self.telemetry.log_subtask(
                    SubtaskTelemetry(
                        event="subtask",
                        ts_ms=now_ms(),
                        run_id=self.run_id,
                        job_id=job_id,
                        subtask_id=a.subtask_id,
                        task_type=s.task_type.value,
                        difficulty=float(s.difficulty),
                        estimated_tokens=int(s.estimated_tokens),
                        assigned_agent=a.agent,
                        agent_id=f"{s.task_type.value}|{a.agent}",
                        routing_mode=a.mode,
                        estimated_performance=float(a.estimated_performance),
                        estimated_token_cost=float(a.estimated_token_cost),
                        estimated_switch_cost_norm=float(a.estimated_switch_cost),
                        swapped_in_for_job=bool(r.swapped_in_for_job),
                        actual_latency_ms=r.latency_ms,
                        input_tokens=r.prompt_tokens,
                        output_tokens=r.completion_tokens,
                        judge_score=ev.judge_score if ev else None,
                        benchmark_name=benchmark_name,
                        benchmark_score=None,
                        success=bool(r.success),
                        error=r.error,
                        output=r.output,
                    )
                )

            plan_step_index += 1

        # Benchmark scoring
        computed_job_score: float | None = None
        computed_subtask_scores: dict[str, float] = {}
        if benchmark_name is not None and benchmark_reference:
            try:
                ensure_default_benchmarks_loaded()
                bench = get_benchmark(benchmark_name)
                ex = BenchmarkExample(benchmark=benchmark_name, example_id=job_id, query=query, reference=benchmark_reference or {})
                tmp_artifact = {"results": [asdict(r) for r in results_all], "subtasks": [{"id": s.id, "task_type": s.task_type.value, "description": s.description, "estimated_tokens": s.estimated_tokens, "difficulty": s.difficulty} for s in all_subtasks], "benchmark_reference": benchmark_reference or {}}
                if hasattr(bench, "score_artifact"):
                    scored = bench.score_artifact(example=ex, artifact=tmp_artifact)  # type: ignore[attr-defined]
                    computed_job_score = float(scored.get("score", 0.0))
                    sub_scores = scored.get("subtask_scores")
                    if isinstance(sub_scores, list):
                        refs = (benchmark_reference or {}).get("subtasks") if isinstance(benchmark_reference, dict) else None
                        if isinstance(refs, list) and len(refs) == len(sub_scores):
                            for ref_s, sc in zip(refs, sub_scores):
                                computed_subtask_scores[str(ref_s.get("id"))] = float(sc)
            except Exception:
                pass

        # Recompute model_task_estimates and oracle for first layer only (simplified)
        if all_assignments and all_subtasks:
            model_task_estimates, oracle = _compute_estimates_and_oracle(
                subtasks=all_subtasks,
                agents=self.agents,
                estimator=self.estimator,
                loaded_models=loaded_before,
                lambda_token=self.lambda_token,
                lambda_switch=self.lambda_switch,
                token_scale=self.token_scale,
                switch_t_scale_ms=self.switch_t_scale_ms,
                assignments=all_assignments,
                lp_objective_value=None,
            )
        else:
            model_task_estimates = []
            oracle = {}

        assignment_by_subtask = {a.subtask_id: a.agent for a in all_assignments}
        active_models_final = set()
        for a in all_assignments:
            active_models_final.add(a.agent)
        # Use peak concurrent VRAM (max over plan steps), not sum of distinct models.
        # For DAG execution we swap models between steps; peak = max per-step concurrent load.
        vram_distinct_gb = sum(a.vram_gb for a in self.agents if a.name in active_models_final)
        vram_used = float(peak_vram_gb) if peak_vram_gb > 0 else float(vram_distinct_gb)
        routing_final = RoutingResult(
            assignments=all_assignments,
            active_models=sorted(active_models_final),
            vram_used_gb=vram_used,
            vram_violation=vram_used > self.gpu_vram_gb + 1e-6,
            routing_source="lp" if self.routing_mode == "lp" else "llm_planner",
            loaded_models=sorted(loaded_before),
        )

        self.telemetry.log_job(
            JobTelemetry(
                event="job",
                ts_ms=now_ms(),
                run_id=self.run_id,
                job_id=job_id,
                query=query,
                gpu_name=self.gpu_spec.name if self.gpu_spec else None,
                gpu_uuid=self.gpu_spec.uuid if self.gpu_spec else None,
                benchmark_name=benchmark_name,
                routing_mode=self.routing_mode,
                lambda_token=self.lambda_token,
                lambda_switch=self.lambda_switch,
                gpu_vram_gb=self.gpu_vram_gb,
                vram_used_gb=float(vram_used),
                vram_violation=bool(vram_used > self.gpu_vram_gb + 1e-6),
                loaded_models_before=sorted(loaded_before),
                active_models=sorted(active_models_final),
                active_roles=sorted({s.task_type.value for s in all_subtasks}),
                active_role_agents=sorted({f"{s.task_type.value}|{assignment_by_subtask.get(s.id, '')}" for s in all_subtasks if s.id in assignment_by_subtask}),
                n_distinct_models=len(active_models_final),
                n_role_agents=len({f"{s.task_type.value}|{assignment_by_subtask.get(s.id, '')}" for s in all_subtasks}),
                models_swapped_in=sorted(active_models_final - loaded_before),
                estimated_switch_cost_ms=total_estimated_switch_ms,
                sum_token_cost=0.0,
                job_score=computed_job_score,
                token_scale=self.token_scale,
                switch_t_scale_ms=self.switch_t_scale_ms,
            )
        )

        sum_token_cost = 0.0
        result_by_subtask = {r.subtask_id: r for r in results_all}
        for s in all_subtasks:
            a = next((x for x in all_assignments if x.subtask_id == s.id), None)
            r = result_by_subtask.get(s.id)
            if a is None or r is None:
                continue
            token_used = (r.prompt_tokens + r.completion_tokens) if isinstance(r.prompt_tokens, int) and isinstance(r.completion_tokens, int) else s.estimated_tokens
            agent_cost = agent_by_name.get(a.agent).cost_per_token if agent_by_name.get(a.agent) else 1.0
            sum_token_cost += float(agent_cost) * float(token_used)

        self.telemetry.log_job(
            JobTelemetry(
                event="job_costs",
                ts_ms=now_ms(),
                run_id=self.run_id,
                job_id=job_id,
                query=query,
                gpu_name=self.gpu_spec.name if self.gpu_spec else None,
                gpu_uuid=self.gpu_spec.uuid if self.gpu_spec else None,
                benchmark_name=benchmark_name,
                routing_mode=self.routing_mode,
                lambda_token=self.lambda_token,
                lambda_switch=self.lambda_switch,
                gpu_vram_gb=self.gpu_vram_gb,
                vram_used_gb=float(vram_used),
                vram_violation=bool(vram_used > self.gpu_vram_gb + 1e-6),
                loaded_models_before=sorted(loaded_before),
                active_models=sorted(active_models_final),
                active_roles=sorted({s.task_type.value for s in all_subtasks}),
                active_role_agents=sorted({f"{s.task_type.value}|{assignment_by_subtask.get(s.id, '')}" for s in all_subtasks}),
                n_distinct_models=len(active_models_final),
                n_role_agents=len({f"{s.task_type.value}|{assignment_by_subtask.get(s.id, '')}" for s in all_subtasks}),
                models_swapped_in=sorted(active_models_final - loaded_before),
                estimated_switch_cost_ms=total_estimated_switch_ms,
                sum_token_cost=float(sum_token_cost),
                job_score=computed_job_score,
                token_scale=self.token_scale,
                switch_t_scale_ms=self.switch_t_scale_ms,
            )
        )

        n_edges = len([e for n in task_dag.nodes.values() for p in n.depends_on for e in [(p, n.id)] if p in task_dag.nodes])
        self.telemetry.log_job_dag_summary(
            {
                "ts_ms": now_ms(),
                "run_id": self.run_id,
                "job_id": job_id,
                "benchmark_name": benchmark_name,
                "routing_mode": self.routing_mode,
                "is_dag": True,
                "mpc": h_depth > 0,
                "horizon_depth": h_depth,
                "n_nodes": len(task_dag.nodes),
                "n_edges": n_edges,
                "n_layers": len(layers),
                "n_plan_steps": plan_step_index,
                "vram_peak_gb": round(vram_used, 3),
                "vram_distinct_gb": round(vram_distinct_gb, 3),
                "total_latency_ms": sum(r.latency_ms or 0 for r in results_all),
                "total_token_cost": sum_token_cost,
                "total_switch_cost_est": total_estimated_switch_ms,
                "avg_subtask_score": float(sum(computed_subtask_scores.values()) / len(computed_subtask_scores)) if computed_subtask_scores else None,
                "job_success": all(r.success for r in results_all) if results_all else False,
            }
        )

        total_latency = sum(r.latency_ms or 0 for r in results_all) if results_all else None
        total_prompt = sum(r.prompt_tokens or 0 for r in results_all) if results_all else None
        total_completion = sum(r.completion_tokens or 0 for r in results_all) if results_all else None
        eval_by_id_final = {e.subtask_id: e for e in evals_all}
        judge_scores = [e.judge_score for e in evals_all if isinstance(e.judge_score, (int, float))]
        avg_judge = float(sum(judge_scores) / len(judge_scores)) if judge_scores else None
        success_rate = float(sum(1.0 if r.success else 0.0 for r in results_all) / len(results_all)) if results_all else 0.0

        plan = [{"id": s.id, "task_type": s.task_type.value, "description": s.description, "estimated_tokens": s.estimated_tokens, "difficulty": float(s.difficulty)} for s in all_subtasks]
        task_to_model = {a.subtask_id: a.agent for a in all_assignments}

        summary = JobSummary(
            run_id=self.run_id,
            job_id=job_id,
            routing_mode=self.routing_mode,
            query=query,
            final_answer=None,
            benchmark_name=benchmark_name,
            job_score=computed_job_score,
            active_models=sorted(active_models_final),
            models_swapped_in=sorted(active_models_final - loaded_before),
            estimated_switch_cost_ms=total_estimated_switch_ms,
            vram_used_gb=float(vram_used),
            vram_violation=bool(vram_used > self.gpu_vram_gb + 1e-6),
            total_latency_ms=total_latency,
            total_prompt_tokens=int(total_prompt) if total_prompt is not None else None,
            total_completion_tokens=int(total_completion) if total_completion is not None else None,
            avg_judge_score=avg_judge,
            success_rate=success_rate,
            plan=plan,
            task_to_model=task_to_model,
            loaded_models=sorted(loaded_before),
            model_task_estimates=model_task_estimates,
            oracle=oracle,
        )

        artifacts_dir = self.run_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        (artifacts_dir / f"{job_id}.json").write_text(
            json.dumps(
                {
                    "run_id": self.run_id,
                    "job_id": job_id,
                    "routing_mode": self.routing_mode,
                    "routing_source": getattr(routing_final, "routing_source", "unknown"),
                    "query": query,
                    "benchmark_name": benchmark_name,
                    "benchmark_reference": benchmark_reference,
                    "subtasks": [{"id": s.id, "task_type": s.task_type.value, "description": s.description, "estimated_tokens": s.estimated_tokens, "difficulty": s.difficulty} for s in all_subtasks],
                    "assignments": [asdict(a) for a in all_assignments],
                    "results": [asdict(r) for r in results_all],
                    "job_score": computed_job_score,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (self.run_dir / f"summary_{job_id}.json").write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
        (self.run_dir / "summary.json").write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")

        self.loaded_models = active_models_final
        return summary

