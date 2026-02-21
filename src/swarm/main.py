from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from rich.console import Console

from .agent_registry import load_agents, save_agents_json, save_default_agents_json
from .model_manager import OllamaModelManager, normalize_required_models
from .bench_runner import BenchConfig, run_bench
from .experiment_runner import ExperimentConfig, run_experiment
from .orchestrator import SwarmOrchestrator
from .telemetry import default_run_id, make_run_dir
from .presets import preset_auto_fast, preset_fast_tiny, preset_heavy


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="swarm", description="Ollama swarm orchestrator (routing: lp vs llm).")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run one or more jobs (sequentially)")
    run.add_argument("--query", action="append", help="User query / job prompt (repeatable)")
    run.add_argument("--queries_file", default=None, help="Path to a newline-delimited file of queries")
    run.add_argument("--job_id_prefix", default="job", help="Job id prefix for batch runs")
    run.add_argument("--run_id", default=None, help="Run id (defaults to timestamp)")

    run.add_argument("--routing", choices=["lp", "llm"], default="llm", help="Routing mode (lp=CVXPY, llm=planner)")
    run.add_argument("--lambda_token", type=float, default=0.5)
    run.add_argument("--lambda_switch", type=float, default=0.2)
    run.add_argument(
        "--gpu_vram_gb",
        type=float,
        default=0.0,
        help="GPU VRAM capacity (GB). If <= 0, auto-detect from system when possible.",
    )

    run.add_argument("--token_scale", type=float, default=1000.0, help="Token-cost normalization scale for objective")
    run.add_argument("--switch_t_scale_ms", type=float, default=1500.0, help="Switch-cost normalization scale (ms)")

    run.add_argument("--ollama_base_url", default="http://localhost:11434")
    run.add_argument("--agents_file", default=None, help="Path to JSON agent registry file")
    run.add_argument("--ensure_models", action="store_true", help="Auto-pull any missing Ollama models for agents")
    run.add_argument("--write_default_agents", default=None, help="Write default agents JSON to this path and exit")
    run.add_argument("--decomposer_model", default="llama3.1:8b")
    run.add_argument("--planner_model", default="llama3.1:8b")
    run.add_argument("--judge_model", default="llama3.1:8b")
    run.add_argument("--planner_timeout_s", type=float, default=120.0)
    run.add_argument("--dry_run", action="store_true", help="No Ollama calls; generates stub outputs")
    run.add_argument("--judge", action="store_true", help="Use an LLM judge to score outputs (extra calls)")

    run.add_argument("--runs_dir", default="runs", help="Base output dir for telemetry")

    models = sub.add_parser("models", help="List/ensure Ollama models")
    models.add_argument("--ollama_base_url", default="http://localhost:11434")
    models.add_argument("--ensure", action="store_true", help="Pull missing models")
    models.add_argument("--models", action="append", help="Model tag to require (repeatable)")

    write_agents = sub.add_parser("write-agents", help="Write agent registry preset JSON")
    write_agents.add_argument("--preset", choices=["fast", "heavy", "auto_fast"], required=True)
    write_agents.add_argument("--out", required=True, help="Output JSON path")
    write_agents.add_argument("--ollama_base_url", default="http://localhost:11434")
    write_agents.add_argument("--k", type=int, default=3, help="For auto_fast: number of models to include")

    bench = sub.add_parser("bench", help="Run open-source benchmarks (ON vs OFF)")
    bench.add_argument("--benchmark", action="append", default=[], help="Benchmark name (repeatable)")
    bench.add_argument("--data_dir", default=str(Path.home() / ".cache" / "swarm_data"), help="External cache dir for datasets")
    bench.add_argument("--limit", type=int, default=20, help="Examples per benchmark")
    bench.add_argument("--seed", type=int, default=0)
    bench.add_argument("--compare", choices=["lp", "llm", "both"], default="both", help="Run routing lp/llm or both")
    bench.add_argument("--lambda_token", type=float, default=0.5)
    bench.add_argument("--lambda_switch", type=float, default=0.2)
    bench.add_argument("--gpu_vram_gb", type=float, default=0.0, help="If <=0, auto-detect VRAM when possible")
    bench.add_argument("--token_scale", type=float, default=1000.0)
    bench.add_argument("--switch_t_scale_ms", type=float, default=1500.0)
    bench.add_argument("--ollama_base_url", default="http://localhost:11434")
    bench.add_argument("--agents_file", default=None)
    bench.add_argument("--decomposer_model", default="llama3.1:8b")
    bench.add_argument("--planner_model", default="llama3.1:8b")
    bench.add_argument("--judge_model", default="llama3.1:8b")
    bench.add_argument("--planner_timeout_s", type=float, default=120.0)
    bench.add_argument("--ensure_models", action="store_true")
    bench.add_argument("--judge", action="store_true")
    bench.add_argument("--dry_run", action="store_true")
    bench.add_argument("--code_eval", action="store_true", help="Run coding benchmark tests in subprocesses (execs untrusted code)")
    bench.add_argument("--mix", choices=["grouped", "interleave"], default="grouped")
    bench.add_argument("--output_dir", default="bench_runs")

    exp = sub.add_parser("experiment", help="Run benchmark sweeps (lambda_switch, VRAM) and suites")
    exp.add_argument("--suite", choices=["workflow_sweep", "code_math_sweep"], default=None)
    exp.add_argument("--benchmark", action="append", default=[], help="Benchmark name (repeatable)")
    exp.add_argument("--data_dir", default=str(Path.home() / ".cache" / "swarm_data"))
    exp.add_argument("--limit", type=int, default=20)
    exp.add_argument("--seed", type=int, default=0)
    exp.add_argument("--compare", choices=["lp", "llm", "both"], default="both")
    exp.add_argument("--lambda_token", type=float, default=0.5)
    exp.add_argument("--lambda_switches", default="0.0,0.2,0.5", help="Comma-separated list")
    exp.add_argument("--gpu_vrams", default="0.0", help="Comma-separated list (0 uses auto-detect)")
    exp.add_argument("--token_scale", type=float, default=1000.0)
    exp.add_argument("--switch_t_scale_ms", type=float, default=1500.0)
    exp.add_argument("--ollama_base_url", default="http://localhost:11434")
    exp.add_argument("--agents_file", default=None)
    exp.add_argument("--decomposer_model", default="llama3.1:8b")
    exp.add_argument("--planner_model", default="llama3.1:8b")
    exp.add_argument("--judge_model", default="llama3.1:8b")
    exp.add_argument("--planner_timeout_s", type=float, default=120.0)
    exp.add_argument("--ensure_models", action="store_true")
    exp.add_argument("--judge", action="store_true")
    exp.add_argument("--dry_run", action="store_true")
    exp.add_argument("--code_eval", action="store_true")
    exp.add_argument("--mix", choices=["grouped", "interleave"], default="interleave")
    exp.add_argument("--output_dir", default="experiments")
    exp.add_argument("--exp_id", default=None)

    return p


async def _run(args: argparse.Namespace) -> int:
    console = Console()
    run_id = args.run_id or default_run_id()
    run_dir = make_run_dir(args.runs_dir, run_id=run_id)

    if args.write_default_agents:
        save_default_agents_json(Path(args.write_default_agents))
        console.print(f"Wrote default agents to {args.write_default_agents}")
        return 0

    queries: list[str] = []
    if args.query:
        queries.extend([q for q in args.query if q and q.strip()])
    if args.queries_file:
        path = Path(args.queries_file)
        text = path.read_text(encoding="utf-8")
        queries.extend([line.strip() for line in text.splitlines() if line.strip()])
    if not queries:
        raise SystemExit("Provide at least one --query or a --queries_file.")

    agents = load_agents(Path(args.agents_file)) if args.agents_file else load_agents(None)

    if args.ensure_models and not args.dry_run:
        mgr = OllamaModelManager(base_url=args.ollama_base_url)
        required = normalize_required_models([a.name for a in agents], extra=[args.decomposer_model, args.planner_model] + ([args.judge_model] if args.judge else []))
        missing = await mgr.missing_models(required)
        if missing:
            console.print(f"[bold]Pulling missing models[/bold]: {missing}")
            mgr.pull_models_cli(missing)

    orch = SwarmOrchestrator(
        run_id=run_id,
        run_dir=Path(run_dir),
        routing_mode=args.routing,
        lambda_token=args.lambda_token,
        lambda_switch=args.lambda_switch,
        gpu_vram_gb=args.gpu_vram_gb,
        token_scale=args.token_scale,
        switch_t_scale_ms=args.switch_t_scale_ms,
        dry_run=args.dry_run,
        judge=args.judge,
        ollama_base_url=args.ollama_base_url,
        agents=agents,
        decomposer_model=args.decomposer_model,
        planner_model=args.planner_model,
        judge_model=args.judge_model,
        planner_timeout_s=args.planner_timeout_s,
    )

    summaries = []
    for idx, q in enumerate(queries, start=1):
        job_id = f"{args.job_id_prefix}-{idx}"
        summary = await orch.run_job(query=q, job_id=job_id)
        summaries.append(summary)

    console.print("[bold]Done[/bold]")
    for s in summaries:
        console.print(s)
    console.print(f"Telemetry: {str(Path(run_dir) / 'telemetry.jsonl')}")
    console.print(f"Summary:   {str(Path(run_dir) / 'summary.json')}")
    return 0


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "run":
        raise SystemExit(asyncio.run(_run(args)))
    if args.command == "models":
        async def _models() -> int:
            console = Console()
            mgr = OllamaModelManager(base_url=args.ollama_base_url)
            required = normalize_required_models(args.models or [])
            missing = await mgr.missing_models(required) if required else []
            if args.ensure and missing:
                console.print(f"[bold]Pulling missing models[/bold]: {missing}")
                mgr.pull_models_cli(missing)
            models = await mgr.list_models()
            console.print([m.name for m in models])
            if required:
                console.print({"required": required, "missing": missing})
            return 0

        raise SystemExit(asyncio.run(_models()))
    if args.command == "write-agents":
        async def _write_agents() -> int:
            console = Console()
            out = Path(args.out)
            if args.preset == "fast":
                agents = preset_fast_tiny()
                save_agents_json(agents, out)
                console.print(f"Wrote fast preset to {out}")
                console.print("Suggested pulls: tinyllama:latest, gemma2:2b, phi3:mini")
                return 0
            if args.preset == "heavy":
                agents = preset_heavy()
                save_agents_json(agents, out)
                console.print(f"Wrote heavy preset to {out}")
                console.print("Suggested pulls: gemma2:2b, qwen2.5-coder:7b, llama3.1:8b, phi4:14b")
                return 0
            mgr = OllamaModelManager(base_url=args.ollama_base_url)
            installed = await mgr.list_models()
            agents = preset_auto_fast(installed, k=int(args.k))
            save_agents_json(agents, out)
            console.print(f"Wrote auto_fast preset to {out}")
            console.print([a.name for a in agents])
            return 0

        raise SystemExit(asyncio.run(_write_agents()))
    if args.command == "bench":
        async def _bench() -> int:
            console = Console()
            cfg = BenchConfig(
                benchmarks=args.benchmark,
                data_dir=args.data_dir,
                limit=args.limit,
                seed=args.seed,
                mix=args.mix,
                compare=args.compare,
                lambda_token=args.lambda_token,
                lambda_switch=args.lambda_switch,
                gpu_vram_gb=args.gpu_vram_gb,
                token_scale=args.token_scale,
                switch_t_scale_ms=args.switch_t_scale_ms,
                ollama_base_url=args.ollama_base_url,
                agents_file=args.agents_file,
                decomposer_model=args.decomposer_model,
                planner_model=args.planner_model,
                judge_model=args.judge_model,
                planner_timeout_s=args.planner_timeout_s,
                ensure_models=args.ensure_models,
                judge=args.judge,
                dry_run=args.dry_run,
                code_eval=args.code_eval,
                output_dir=args.output_dir,
                bench_id=None,
            )
            return await run_bench(cfg, console=console)

        raise SystemExit(asyncio.run(_bench()))
    if args.command == "experiment":
        async def _exp() -> int:
            console = Console()
            lambda_switches = [float(x.strip()) for x in str(args.lambda_switches).split(",") if x.strip()]
            gpu_vrams = [float(x.strip()) for x in str(args.gpu_vrams).split(",") if x.strip()]
            base = BenchConfig(
                benchmarks=args.benchmark or (["workflowbench"] if args.suite == "workflow_sweep" else ["code_math_mix"]),
                data_dir=args.data_dir,
                limit=args.limit,
                seed=args.seed,
                mix=args.mix,
                compare=args.compare,
                lambda_token=args.lambda_token,
                lambda_switch=lambda_switches[0] if lambda_switches else 0.2,
                gpu_vram_gb=gpu_vrams[0] if gpu_vrams else 0.0,
                token_scale=args.token_scale,
                switch_t_scale_ms=args.switch_t_scale_ms,
                ollama_base_url=args.ollama_base_url,
                agents_file=args.agents_file,
                decomposer_model=args.decomposer_model,
                planner_model=args.planner_model,
                judge_model=args.judge_model,
                planner_timeout_s=args.planner_timeout_s,
                ensure_models=args.ensure_models,
                judge=args.judge,
                dry_run=args.dry_run,
                code_eval=args.code_eval,
                output_dir="bench_runs",
                bench_id=None,
            )
            cfg = ExperimentConfig(
                exp_id=args.exp_id,
                output_dir=args.output_dir,
                suite=args.suite,
                lambda_switches=lambda_switches or [0.0, 0.2, 0.5],
                gpu_vrams=gpu_vrams or [0.0],
                base=base,
            )
            return await run_experiment(cfg, console=console)

        raise SystemExit(asyncio.run(_exp()))

