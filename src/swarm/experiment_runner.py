from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from rich.console import Console

from .bench_runner import BenchConfig, run_bench
from .telemetry import default_run_id


@dataclass(frozen=True)
class ExperimentConfig:
    exp_id: str | None
    output_dir: str
    suite: str | None  # e.g. "switch_stress"
    lambda_switches: list[float]
    gpu_vrams: list[float]
    base: BenchConfig


def _suite_defaults(suite: str) -> dict:
    if suite == "workflow_sweep":
        return {"benchmarks": ["workflowbench"], "mix": "grouped"}
    if suite == "code_math_sweep":
        return {"benchmarks": ["code_math_mix"], "mix": "interleave"}
    return {}


async def run_experiment(cfg: ExperimentConfig, *, console: Console) -> int:
    exp_id = cfg.exp_id or default_run_id()
    exp_dir = Path(cfg.output_dir) / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Apply suite overrides.
    base = cfg.base
    if cfg.suite:
        overrides = _suite_defaults(cfg.suite)
        if "benchmarks" in overrides:
            base = replace(base, benchmarks=overrides["benchmarks"])
        if "mix" in overrides:
            base = replace(base, mix=overrides["mix"])

    index: dict = {
        "exp_id": exp_id,
        "suite": cfg.suite,
        "lambda_switches": cfg.lambda_switches,
        "gpu_vrams": cfg.gpu_vrams,
        "runs": [],
    }

    for ls in cfg.lambda_switches:
        for vram in cfg.gpu_vrams:
            run_id = f"{exp_id}-ls{ls}-vram{vram}"
            console.print(f"Running {run_id}")
            run_cfg = replace(
                base,
                lambda_switch=float(ls),
                gpu_vram_gb=float(vram),
                output_dir=str(exp_dir / "bench_runs"),
                bench_id=run_id,
            )
            await run_bench(run_cfg, console=console)
            report_path = Path(run_cfg.output_dir) / run_id / "report.json"
            index["runs"].append(
                {
                    "bench_id": run_id,
                    "lambda_switch": float(ls),
                    "gpu_vram_gb": float(vram),
                    "report": str(report_path),
                }
            )

    (exp_dir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    console.print(f"Wrote experiment index to {exp_dir / 'index.json'}")
    return 0

