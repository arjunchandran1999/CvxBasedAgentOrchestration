from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .benchmarks.registry import BenchmarkExample, ensure_default_benchmarks_loaded, get as get_bench


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_examples(bench_dir: Path) -> list[BenchmarkExample]:
    examples_path = bench_dir / "examples.jsonl"
    examples: list[BenchmarkExample] = []
    for line in examples_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        examples.append(
            BenchmarkExample(
                benchmark=d["benchmark"],
                example_id=d["example_id"],
                query=d["query"],
                reference=d["reference"],
            )
        )
    return examples


def _load_job_events(run_dir: Path) -> dict[str, dict[str, Any]]:
    """
    job_id -> job telemetry dict
    """
    path = run_dir / "telemetry.jsonl"
    jobs: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return jobs
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("event") in {"job", "job_costs"}:
            # keep last-seen event as canonical
            jobs[str(d["job_id"])] = d
    return jobs


def _load_subtask_events(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "telemetry.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("event") == "subtask":
            out.append(d)
    return out


def aggregate_reports(*, bench_dir: Path, run_dirs: dict[str, Path], benchmarks: list[str]) -> dict[str, Any]:
    """
    Aggregates ON/OFF benchmark scores + telemetry into JSON.
    Also writes a per-example CSV to `bench_dir/report.csv`.
    """
    ensure_default_benchmarks_loaded()
    examples = _load_examples(bench_dir)
    benches = {b: get_bench(b) for b in set([e.benchmark for e in examples])}

    report: dict[str, Any] = {
        "bench_dir": str(bench_dir),
        "n_examples": len(examples),
        "modes": {},
    }

    rows: list[dict[str, Any]] = []
    output_lines: list[dict[str, Any]] = []

    for mode, run_dir in run_dirs.items():
        artifacts = run_dir / "artifacts"
        job_events = _load_job_events(run_dir)
        subtask_events = _load_subtask_events(run_dir)

        scores: list[float] = []
        per_bench_scores: dict[str, list[float]] = {}
        swap_counts: list[float] = []
        switch_ms: list[float] = []
        active_model_counts: list[float] = []
        vram_used: list[float] = []

        for idx, ex in enumerate(examples, start=1):
            job_id = f"{ex.benchmark}-{idx}"
            art_path = artifacts / f"{job_id}.json"
            if not art_path.exists():
                continue
            art = _read_json(art_path)
            bench = benches[ex.benchmark]
            if hasattr(bench, "score_artifact"):
                s = bench.score_artifact(example=ex, artifact=art)  # type: ignore[attr-defined]
            else:
                final_answer = art.get("final_answer")
                s = bench.score(example=ex, final_answer=final_answer)
            val = float(s.get("score", 0.0))
            scores.append(val)
            per_bench_scores.setdefault(ex.benchmark, []).append(val)

            jt = job_events.get(job_id) or {}
            swaps = float(len(jt.get("models_swapped_in") or []))
            swap_counts.append(swaps)
            switch_ms.append(float(jt.get("estimated_switch_cost_ms") or 0.0))
            active_model_counts.append(float(len(jt.get("active_models") or [])))
            vram_used.append(float(jt.get("vram_used_gb") or 0.0))

            rows.append(
                {
                    "routing_mode": mode,
                    "benchmark": ex.benchmark,
                    "example_id": ex.example_id,
                    "job_id": job_id,
                    "metric": s.get("metric"),
                    "score": val,
                    "active_models": len(jt.get("active_models") or []),
                    "active_roles": len(jt.get("active_roles") or []),
                    "active_role_agents": len(jt.get("active_role_agents") or []),
                    "models_swapped_in": len(jt.get("models_swapped_in") or []),
                    "estimated_switch_cost_ms": float(jt.get("estimated_switch_cost_ms") or 0.0),
                    "vram_used_gb": float(jt.get("vram_used_gb") or 0.0),
                    "vram_violation": bool(jt.get("vram_violation") or False),
                    "sum_token_cost": float(jt.get("sum_token_cost") or 0.0),
                    "gpu_name": jt.get("gpu_name"),
                }
            )

            # Collect subtask outputs for outputs.jsonl (workflowbench etc.)
            sub_scores = s.get("subtask_scores")
            if not isinstance(sub_scores, list):
                sub_scores = []
            results_list = art.get("results") or []
            ref_subs = (ex.reference or {}).get("subtasks", []) if isinstance(ex.reference, dict) else []
            for res in results_list:
                if isinstance(res, dict) and "subtask_id" in res:
                    sid = res.get("subtask_id", "")
                    out_text = res.get("output", "")
                    expected = next(
                        (rs.get("expected") for rs in ref_subs if str(rs.get("id")) == str(sid)),
                        None,
                    )
                    sub_score = None
                    for i, rs in enumerate(ref_subs):
                        if str(rs.get("id")) == str(sid) and i < len(sub_scores):
                            sub_score = sub_scores[i]
                            break
                    output_lines.append({
                        "routing_mode": mode,
                        "job_id": job_id,
                        "example_id": ex.example_id,
                        "subtask_id": sid,
                        "output": out_text,
                        "expected": expected,
                        "benchmark_score": sub_score,
                        "agent": res.get("agent"),
                    })

        report["modes"][mode] = {
            "n_scored": len(scores),
            "avg_score": (sum(scores) / len(scores)) if scores else None,
            "by_benchmark": {k: (sum(v) / len(v) if v else None) for k, v in per_bench_scores.items()},
            "avg_models_swapped_in": (sum(swap_counts) / len(swap_counts)) if swap_counts else None,
            "avg_estimated_switch_cost_ms": (sum(switch_ms) / len(switch_ms)) if switch_ms else None,
            "avg_active_model_count": (sum(active_model_counts) / len(active_model_counts)) if active_model_counts else None,
            "avg_vram_used_gb": (sum(vram_used) / len(vram_used)) if vram_used else None,
            "run_dir": str(run_dir),
        }

    # Write CSV
    if rows:
        csv_path = bench_dir / "report.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        report["report_csv"] = str(csv_path)

    # Write outputs.jsonl (subtask outputs for workflowbench etc.)
    if output_lines:
        out_path = bench_dir / "outputs.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for line in output_lines:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        report["outputs_jsonl"] = str(out_path)

    return report

