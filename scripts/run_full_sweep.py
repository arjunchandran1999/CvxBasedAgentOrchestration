#!/usr/bin/env python3
"""
Master experiment script: runs all key benchmarks and sweeps into a timestamped directory.

Creates: runs/<timestamp>_full_sweep/
  - workflowbench/... (VRAM 8GB and 24GB, horizon_depth 1 and 3)
  - code_math_mix/... (lambda_switch sweep, grouped and interleave)
  - agentic_bench/... (if available)
  - meta_report.json (aggregated Pareto data from all report.json files)

Usage:
  python scripts/run_full_sweep.py
  # or
  ./scripts/run_experiments.sh  # if wrapped
"""

from __future__ import annotations

import datetime
import json
import os
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], cwd: Path | None = None, dry_run: bool = False) -> None:
    print(f"  $ {' '.join(cmd)}")
    if dry_run:
        print("  [dry run, skipped]")
        return
    r = subprocess.run(cmd, cwd=cwd or Path.cwd(), check=False)
    if r.returncode != 0:
        print(f"  [FAILED exit {r.returncode}]")
        sys.exit(r.returncode)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Run full benchmark sweep")
    ap.add_argument("--dry_run", action="store_true", help="Skip actual runs, only create dir structure")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    os.chdir(repo_root)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    root_dir = Path("runs") / f"{timestamp}_full_sweep"
    root_dir.mkdir(parents=True, exist_ok=True)
    print(f"Full sweep root: {root_dir}")

    agents_heavy = "configs/agents_heavy.json"
    agents_fast = "configs/agents_fast.json"
    if not (repo_root / agents_heavy).exists():
        agents_heavy = "configs/agents_fast.json"
    if not (repo_root / agents_fast).exists():
        agents_fast = agents_heavy

    dry = getattr(args, "dry_run", False)

    # --- WorkflowBench (DAG, VRAM stress) ---
    print("\n=== WorkflowBench ===")
    for vram in [8, 24]:
        for hd in [1, 3]:
            out = root_dir / "workflowbench" / f"vram{vram}_hd{hd}"
            _run([
                "swarm", "bench",
                "--benchmark", "workflowbench",
                "--agents_file", str(repo_root / agents_heavy),
                "--gpu_vram_gb", str(vram),
                "--compare", "both",
                "--limit", "3",
                "--horizon_depth", str(hd),
                "--output_dir", str(out),
            ], dry_run=dry)

    # --- CodeMathMix (flat, lambda sweep) ---
    print("\n=== CodeMathMix ===")
    for lam_sw in [0.0, 0.2, 0.5, 1.0]:
        out = root_dir / "code_math_mix" / f"ls{lam_sw}"
        _run([
            "swarm", "bench",
            "--benchmark", "code_math_mix",
            "--agents_file", str(repo_root / agents_fast),
            "--gpu_vram_gb", "8",
            "--lambda_switch", str(lam_sw),
            "--compare", "both",
            "--limit", "4",
            "--mix", "interleave",
            "--output_dir", str(out),
        ], dry_run=dry)

    for mix in ["grouped", "interleave"]:
        out = root_dir / "code_math_mix" / f"mix_{mix}"
        _run([
            "swarm", "bench",
            "--benchmark", "code_math_mix",
            "--agents_file", str(repo_root / agents_fast),
            "--gpu_vram_gb", "8",
            "--lambda_switch", "0.5",
            "--compare", "both",
            "--limit", "4",
            "--mix", mix,
            "--output_dir", str(out),
        ], dry_run=dry)

    # --- AgenticBench (if available) ---
    try:
        print("\n=== AgenticBench ===")
        out = root_dir / "agentic_bench" / "vram16_hd2"
        _run([
            "swarm", "bench",
            "--benchmark", "agentic_bench",
            "--agents_file", str(repo_root / agents_heavy),
            "--gpu_vram_gb", "16",
            "--compare", "both",
            "--limit", "1",
            "--horizon_depth", "2",
            "--output_dir", str(out),
        ], dry_run=dry)
    except Exception:
        print("  AgenticBench skipped (not registered or error)")

    # --- Aggregate meta_report.json ---
    reports = list(root_dir.rglob("report.json"))
    entries: list[dict] = []
    for rp in reports:
        try:
            data = json.loads(rp.read_text(encoding="utf-8"))
            bench_dir = rp.parent
            config_path = bench_dir / "bench_config.json"
            config: dict = {}
            if config_path.exists():
                config = json.loads(config_path.read_text(encoding="utf-8"))
            for mode, mode_data in (data.get("modes") or {}).items():
                e: dict = {
                    "bench_dir": str(bench_dir),
                    "benchmark_name": list(config.get("benchmarks", ["unknown"]))[0] if config.get("benchmarks") else "unknown",
                    "routing_mode": mode,
                    "lambda_token": config.get("lambda_token", 0.5),
                    "lambda_switch": config.get("lambda_switch", 0.2),
                    "horizon_depth": config.get("horizon_depth", 1),
                    "gpu_vram_gb": config.get("gpu_vram_gb", 0),
                    "quality_estimator_type": "static",
                    "mean_score": mode_data.get("avg_score"),
                    "mean_token_cost": None,
                    "mean_switch_cost_est": mode_data.get("avg_estimated_switch_cost_ms"),
                    "mean_peak_vram_used_gb": mode_data.get("avg_vram_used_gb"),
                }
                # Get mean token cost from pareto_data if available
                pareto = data.get("pareto_data") or {}
                for _bn, points in pareto.items():
                    for p in points:
                        if p.get("routing_mode") == mode:
                            e["mean_token_cost"] = p.get("mean_token_cost")
                            break
                entries.append(e)
        except Exception as ex:
            print(f"  Skip {rp}: {ex}")

    meta = {
        "runs_root": str(root_dir),
        "timestamp": timestamp,
        "reports": entries,
    }
    (root_dir / "meta_report.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nWrote meta_report.json ({len(entries)} entries) to {root_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
