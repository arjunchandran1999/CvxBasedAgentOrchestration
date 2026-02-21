from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .agents import Agent, default_agents
from .estimator import QualityEstimator
from .executor import SubtaskResult, execute
from .evaluator import Evaluation, evaluate
from .ollama_client import OllamaClient
from .optimizer import OptimizerConfig, SwarmOptimizer
from .planner_router import llm_assign
from .routing import Assignment, RoutingResult
from .tasks import Subtask, decompose
from typing import Literal

from .benchmarks.registry import BenchmarkExample, ensure_default_benchmarks_loaded, get as get_benchmark

from .telemetry import JobTelemetry, OrchestrationTelemetry, SubtaskTelemetry, TelemetryLogger, now_ms
from .gpu import GpuSpec, detect_gpu, get_effective_vram_gb


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
        else:
            subtasks = await decompose(query, self.ollama, planner_model=self.decomposer_model)

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

