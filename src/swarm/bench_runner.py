from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from rich.console import Console

from .agent_registry import load_agents
from .model_manager import OllamaModelManager, normalize_required_models
from .orchestrator import SwarmOrchestrator
from .telemetry import default_run_id, make_run_dir
from .benchmarks.registry import BenchmarkExample, ensure_default_benchmarks_loaded, get as get_bench, list_benchmarks
from .reporting import aggregate_reports


@dataclass(frozen=True)
class BenchConfig:
    benchmarks: list[str]
    data_dir: str
    limit: int
    seed: int
    mix: str  # "grouped" | "interleave"
    compare: str  # "lp" | "llm" | "both"
    lambda_token: float
    lambda_switch: float
    gpu_vram_gb: float
    token_scale: float
    switch_t_scale_ms: float
    ollama_base_url: str
    agents_file: str | None
    decomposer_model: str = "llama3.1:8b"
    planner_model: str = "llama3.1:8b"
    judge_model: str = "llama3.1:8b"
    planner_timeout_s: float = 120.0
    ensure_models: bool = False
    judge: bool = False
    dry_run: bool = False
    code_eval: bool = False
    output_dir: str = "bench_runs"
    bench_id: str | None = None


def _set_hf_cache(data_dir: str) -> None:
    # Keep datasets out of the repo. Datasets will be cached under data_dir.
    os.environ.setdefault("HF_HOME", data_dir)
    os.environ.setdefault("HF_DATASETS_CACHE", str(Path(data_dir) / "datasets"))
    os.environ.setdefault("HF_EVALUATE_CACHE", str(Path(data_dir) / "evaluate"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(data_dir) / "transformers"))


async def run_bench(cfg: BenchConfig, *, console: Console) -> int:
    _set_hf_cache(cfg.data_dir)
    if cfg.code_eval:
        os.environ["SWARM_CODE_EVAL"] = "1"

    ensure_default_benchmarks_loaded()
    if not cfg.benchmarks:
        raise SystemExit(f"Select at least one benchmark. Known: {list_benchmarks()}")

    benches = [get_bench(b) for b in cfg.benchmarks]
    loaded_by_bench: list[list[BenchmarkExample]] = [b.load(data_dir=cfg.data_dir, limit=cfg.limit, seed=cfg.seed) for b in benches]

    all_examples: list[BenchmarkExample] = []
    if cfg.mix == "interleave":
        # Round-robin to amplify model-switching behavior across task types.
        i = 0
        while any(loaded_by_bench):
            for j in range(len(loaded_by_bench)):
                if loaded_by_bench[j]:
                    all_examples.append(loaded_by_bench[j].pop(0))
            i += 1
            if i > 100000:
                break
    else:
        for xs in loaded_by_bench:
            all_examples.extend(xs)

    if not all_examples:
        raise SystemExit("No examples loaded (check benchmark names/limit).")

    # Persist example metadata for reproducibility.
    bench_id = cfg.bench_id or default_run_id()
    bench_dir = Path(cfg.output_dir) / bench_id
    bench_dir.mkdir(parents=True, exist_ok=True)
    (bench_dir / "bench_config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    (bench_dir / "examples.jsonl").write_text(
        "\n".join(json.dumps(asdict(e), ensure_ascii=False) for e in all_examples) + "\n",
        encoding="utf-8",
    )

    agents = load_agents(Path(cfg.agents_file)) if cfg.agents_file else load_agents(None)

    if cfg.ensure_models and not cfg.dry_run:
        mgr = OllamaModelManager(base_url=cfg.ollama_base_url)
        required = normalize_required_models([a.name for a in agents], extra=[cfg.decomposer_model, cfg.planner_model, cfg.judge_model] if cfg.judge_model else [cfg.decomposer_model, cfg.planner_model])
        missing = await mgr.missing_models(required)
        if missing:
            console.print(f"[bold]Pulling missing models[/bold]: {missing}")
            mgr.pull_models_cli(missing)

    modes = [cfg.compare] if cfg.compare != "both" else ["lp", "llm"]
    run_dirs: dict[str, Path] = {}

    for mode in modes:
        run_id = f"{bench_id}-{mode}"
        run_dir = make_run_dir(str(bench_dir / "runs"), run_id=run_id)
        run_dirs[mode] = Path(run_dir)
        orch = SwarmOrchestrator(
            run_id=run_id,
            run_dir=Path(run_dir),
            routing_mode=mode,  # type: ignore[arg-type]
            lambda_token=cfg.lambda_token,
            lambda_switch=cfg.lambda_switch,
            gpu_vram_gb=cfg.gpu_vram_gb,
            token_scale=cfg.token_scale,
            switch_t_scale_ms=cfg.switch_t_scale_ms,
            dry_run=cfg.dry_run,
            judge=cfg.judge,
            ollama_base_url=cfg.ollama_base_url,
            agents=agents,
            decomposer_model=cfg.decomposer_model,
            planner_model=cfg.planner_model,
            judge_model=cfg.judge_model,
            planner_timeout_s=cfg.planner_timeout_s,
        )

        for idx, ex in enumerate(all_examples, start=1):
            job_id = f"{ex.benchmark}-{idx}"
            bench = get_bench(ex.benchmark)
            subtasks_override = None
            if hasattr(bench, "make_subtasks"):
                try:
                    subtasks_override = bench.make_subtasks(example=ex)  # type: ignore[attr-defined]
                except Exception:
                    subtasks_override = None
            await orch.run_job(
                query=ex.query,
                job_id=job_id,
                benchmark_name=ex.benchmark,
                benchmark_reference=ex.reference,
                subtasks_override=subtasks_override,
            )

    # Score by reading artifacts and applying benchmark scorers.
    report = aggregate_reports(
        bench_dir=bench_dir,
        run_dirs=run_dirs,
        benchmarks=cfg.benchmarks,
    )
    (bench_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    console.print(f"Wrote report to {bench_dir / 'report.json'}")
    return 0

