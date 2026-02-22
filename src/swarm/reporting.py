from __future__ import annotations

import csv
import json
import statistics
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


def _load_plan_step_events(run_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """job_id -> list of plan_step events (for DAG runs; LP solver diagnostics)."""
    path = run_dir / "telemetry.jsonl"
    by_job: dict[str, list[dict[str, Any]]] = {}
    if not path.exists():
        return by_job
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("event") == "plan_step":
            jid = str(d.get("job_id", ""))
            if jid:
                by_job.setdefault(jid, []).append(d)
    return by_job


def _aggregate_subtask_metrics_per_job(subtask_events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """job_id -> {total_latency_ms, total_prompt_tokens, total_completion_tokens}."""
    by_job: dict[str, dict[str, float | int]] = {}
    for ev in subtask_events:
        jid = str(ev.get("job_id", ""))
        if not jid:
            continue
        if jid not in by_job:
            by_job[jid] = {"total_latency_ms": 0.0, "total_prompt_tokens": 0, "total_completion_tokens": 0}
        lat = ev.get("actual_latency_ms")
        if isinstance(lat, (int, float)):
            by_job[jid]["total_latency_ms"] += float(lat)
        inp = ev.get("input_tokens")
        if isinstance(inp, int):
            by_job[jid]["total_prompt_tokens"] += inp
        out_tok = ev.get("output_tokens")
        if isinstance(out_tok, int):
            by_job[jid]["total_completion_tokens"] += out_tok
    return by_job


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

    # Pre-load LP artifacts for oracle (LP optimal utility) to compute LLM oracle_gap
    job_oracle_utility: dict[str, float] = {}
    if "lp" in run_dirs:
        lp_artifacts = run_dirs["lp"] / "artifacts"
        for idx, ex in enumerate(examples, start=1):
            job_id = f"{ex.benchmark}-{idx}"
            art_path = lp_artifacts / f"{job_id}.json"
            if art_path.exists():
                art = _read_json(art_path)
                oracle = art.get("oracle") or {}
                lp_obj = oracle.get("lp_objective_value")
                if lp_obj is not None:
                    job_oracle_utility[job_id] = float(lp_obj)

    # Cross-mode job data for jobs_won, routing_divergence
    job_scores: dict[str, dict[str, float]] = {}
    job_vram_violations: dict[str, dict[str, bool]] = {}
    job_assignments: dict[str, dict[str, dict[str, str]]] = {}  # job_id -> mode -> subtask_id -> model
    job_subtask_scores: dict[str, dict[str, dict[str, float]]] = {}  # job_id -> mode -> subtask_id -> score

    for mode, run_dir in run_dirs.items():
        artifacts = run_dir / "artifacts"
        job_events = _load_job_events(run_dir)
        subtask_events = _load_subtask_events(run_dir)
        plan_steps = _load_plan_step_events(run_dir)
        subtask_agg = _aggregate_subtask_metrics_per_job(subtask_events)

        scores: list[float] = []
        per_bench_scores: dict[str, list[float]] = {}
        swap_counts: list[float] = []
        switch_ms: list[float] = []
        active_model_counts: list[float] = []
        vram_used: list[float] = []
        total_latencies: list[float] = []
        total_prompt_tokens_list: list[int] = []
        total_completion_tokens_list: list[int] = []
        score_per_token_costs: list[float] = []
        lp_objectives: list[float] = []
        chosen_utilities: list[float] = []
        lp_solve_times_ms: list[float] = []
        vram_violation_count: int = 0
        oracle_gaps_pct: list[float] = []
        total_cost_ests: list[float] = []
        utility_efficiencies: list[float] = []

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
            vram_used_gb = float(jt.get("vram_used_gb") or 0.0)
            vram_used.append(vram_used_gb)
            gpu_vram = float(jt.get("gpu_vram_gb") or 0.0)

            st_agg = subtask_agg.get(job_id) or {}
            total_latency_ms = st_agg.get("total_latency_ms") or 0.0
            total_prompt_tokens = st_agg.get("total_prompt_tokens") or 0
            total_completion_tokens = st_agg.get("total_completion_tokens") or 0
            total_latencies.append(float(total_latency_ms))
            total_prompt_tokens_list.append(int(total_prompt_tokens))
            total_completion_tokens_list.append(int(total_completion_tokens))

            sum_tok = float(jt.get("sum_token_cost") or 0.0)
            score_per_tok = val / sum_tok if sum_tok > 0 else (val if val == 0 else float("inf"))
            score_per_token_costs.append(score_per_tok)

            oracle = art.get("oracle") or {}
            lp_obj = oracle.get("lp_objective_value")
            chosen_util = oracle.get("chosen_utility")
            if lp_obj is not None:
                lp_objectives.append(float(lp_obj))
            if chosen_util is not None:
                chosen_utilities.append(float(chosen_util))

            lp_solve_total = 0.0
            for ps in plan_steps.get(job_id) or []:
                t = ps.get("lp_solve_time_ms")
                if isinstance(t, (int, float)):
                    lp_solve_total += float(t)
            if lp_solve_total > 0:
                lp_solve_times_ms.append(lp_solve_total)

            vram_util = vram_used_gb / gpu_vram if gpu_vram > 0 else None
            vram_vio = bool(jt.get("vram_violation") or False)
            if vram_vio:
                vram_violation_count += 1

            # Cross-mode tracking
            job_scores.setdefault(job_id, {})[mode] = val
            job_vram_violations.setdefault(job_id, {})[mode] = vram_vio

            # Oracle gap (LLM only): how far below LP optimum
            oracle_util = job_oracle_utility.get(job_id)
            if mode == "llm" and oracle_util is not None and oracle_util > 0 and chosen_util is not None:
                gap_pct = 100.0 * (oracle_util - float(chosen_util)) / oracle_util
                oracle_gaps_pct.append(gap_pct)

            # Total cost estimate (token + switch proxy); k=0.001 converts ms to cost-like units
            switch_ms_val = float(jt.get("estimated_switch_cost_ms") or 0.0)
            total_cost_est = sum_tok + 0.001 * switch_ms_val
            if total_cost_est > 0 and chosen_util is not None:
                utility_efficiencies.append(float(chosen_util) / total_cost_est)
            total_cost_ests.append(total_cost_est)

            # Store assignments and subtask scores for routing_divergence
            assignments = art.get("assignments") or (art.get("orchestration") or {}).get("assignments") or []
            if isinstance(assignments, list):
                assign_map: dict[str, str] = {}
                for a in assignments:
                    if isinstance(a, dict):
                        sid = a.get("subtask_id")
                        model = a.get("model") or a.get("agent")
                        if sid is not None and model is not None:
                            assign_map[str(sid)] = str(model)
                job_assignments.setdefault(job_id, {})[mode] = assign_map
            sub_scores_map: dict[str, float] = {}
            if isinstance(sub_scores := s.get("subtask_scores"), list):
                ref_subs = (ex.reference or {}).get("subtasks", []) if isinstance(ex.reference, dict) else []
                for i, rs in enumerate(ref_subs):
                    if i < len(sub_scores):
                        sub_scores_map[str(rs.get("id", ""))] = float(sub_scores[i])
            if sub_scores_map:
                job_subtask_scores.setdefault(job_id, {})[mode] = sub_scores_map

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
                    "vram_used_gb": vram_used_gb,
                    "vram_violation": bool(jt.get("vram_violation") or False),
                    "sum_token_cost": sum_tok,
                    "gpu_name": jt.get("gpu_name"),
                    "total_latency_ms": total_latency_ms,
                    "total_prompt_tokens": total_prompt_tokens,
                    "total_completion_tokens": total_completion_tokens,
                    "score_per_token_cost": round(score_per_tok, 6) if sum_tok > 0 else None,
                    "vram_utilization": round(vram_util, 4) if vram_util is not None else None,
                    "lp_objective_value": round(float(lp_obj), 4) if lp_obj is not None else None,
                    "chosen_utility": round(float(chosen_util), 4) if chosen_util is not None else None,
                    "total_cost_est": round(total_cost_est, 4),
                    "utility_efficiency": round(chosen_util / total_cost_est, 6) if chosen_util is not None and total_cost_est > 0 else None,
                    "oracle_gap_pct": round(100.0 * (job_oracle_utility[job_id] - float(chosen_util)) / job_oracle_utility[job_id], 2) if mode == "llm" and job_id in job_oracle_utility and job_oracle_utility[job_id] > 0 and chosen_util is not None else None,
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

        mode_data: dict[str, Any] = {
            "n_scored": len(scores),
            "avg_score": (sum(scores) / len(scores)) if scores else None,
            "by_benchmark": {k: (sum(v) / len(v) if v else None) for k, v in per_bench_scores.items()},
            "avg_models_swapped_in": (sum(swap_counts) / len(swap_counts)) if swap_counts else None,
            "avg_estimated_switch_cost_ms": (sum(switch_ms) / len(switch_ms)) if switch_ms else None,
            "avg_active_model_count": (sum(active_model_counts) / len(active_model_counts)) if active_model_counts else None,
            "avg_vram_used_gb": (sum(vram_used) / len(vram_used)) if vram_used else None,
            "run_dir": str(run_dir),
            "vram_violation_rate": (vram_violation_count / len(scores)) if scores else 0.0,
            "vram_violation_count": vram_violation_count,
        }
        if len(scores) >= 2:
            mode_data["score_std"] = round(statistics.stdev(scores), 6)
        if scores:
            mode_data["score_min"] = min(scores)
            mode_data["score_max"] = max(scores)
        if oracle_gaps_pct:
            mode_data["avg_oracle_gap_pct"] = round(sum(oracle_gaps_pct) / len(oracle_gaps_pct), 2)
        if total_cost_ests:
            mode_data["avg_total_cost_est"] = sum(total_cost_ests) / len(total_cost_ests)
        if utility_efficiencies:
            mode_data["avg_utility_efficiency"] = sum(utility_efficiencies) / len(utility_efficiencies)
        if total_latencies:
            mode_data["avg_total_latency_ms"] = sum(total_latencies) / len(total_latencies)
            mode_data["total_latency_ms_sum"] = sum(total_latencies)
        if total_prompt_tokens_list:
            mode_data["avg_total_prompt_tokens"] = sum(total_prompt_tokens_list) / len(total_prompt_tokens_list)
            mode_data["total_prompt_tokens_sum"] = sum(total_prompt_tokens_list)
        if total_completion_tokens_list:
            mode_data["avg_total_completion_tokens"] = sum(total_completion_tokens_list) / len(total_completion_tokens_list)
            mode_data["total_completion_tokens_sum"] = sum(total_completion_tokens_list)
        if score_per_token_costs:
            valid = [x for x in score_per_token_costs if x != float("inf")]
            mode_data["avg_score_per_token_cost"] = sum(valid) / len(valid) if valid else None
        if lp_objectives:
            mode_data["avg_lp_objective_value"] = sum(lp_objectives) / len(lp_objectives)
        if chosen_utilities:
            mode_data["avg_chosen_utility"] = sum(chosen_utilities) / len(chosen_utilities)
        if lp_solve_times_ms:
            mode_data["avg_lp_solve_time_ms"] = sum(lp_solve_times_ms) / len(lp_solve_times_ms)
            mode_data["total_lp_solve_time_ms"] = sum(lp_solve_times_ms)
        report["modes"][mode] = mode_data

    # LP vs LLM comparison (when both modes present)
    if "lp" in report["modes"] and "llm" in report["modes"]:
        lp_m = report["modes"]["lp"]
        llm_m = report["modes"]["llm"]
        comparison: dict[str, Any] = {}
        for key in ["avg_score", "avg_models_swapped_in", "avg_estimated_switch_cost_ms", "avg_vram_used_gb",
                    "avg_total_latency_ms", "avg_score_per_token_cost", "avg_chosen_utility",
                    "avg_utility_efficiency", "avg_oracle_gap_pct"]:
            lp_v = lp_m.get(key)
            llm_v = llm_m.get(key)
            if lp_v is not None and llm_v is not None and isinstance(lp_v, (int, float)) and isinstance(llm_v, (int, float)):
                comparison[f"{key}_lp_minus_llm"] = round(float(lp_v) - float(llm_v), 6)
                if llm_v != 0:
                    comparison[f"{key}_lp_vs_llm_pct"] = round(100.0 * (float(lp_v) - float(llm_v)) / float(llm_v), 2)
        comparison["vram_violation_rate_lp"] = lp_m.get("vram_violation_rate", 0.0)
        comparison["vram_violation_rate_llm"] = llm_m.get("vram_violation_rate", 0.0)
        if llm_m.get("avg_oracle_gap_pct") is not None:
            comparison["llm_avg_oracle_gap_pct"] = llm_m["avg_oracle_gap_pct"]

        # jobs_won: LP wins when lp_score > llm_score for same job
        lp_wins = 0.0
        llm_wins = 0.0
        ties = 0.0
        for jid, mode_scores in job_scores.items():
            lp_s = mode_scores.get("lp")
            llm_s = mode_scores.get("llm")
            if lp_s is not None and llm_s is not None:
                if lp_s > llm_s:
                    lp_wins += 1.0
                elif llm_s > lp_s:
                    llm_wins += 1.0
                else:
                    ties += 1.0
        comparison["jobs_lp_won"] = int(lp_wins)
        comparison["jobs_llm_won"] = int(llm_wins)
        comparison["jobs_tied"] = int(ties)
        comparison["jobs_won_summary"] = f"LP {int(lp_wins)} / LLM {int(llm_wins)} / tied {int(ties)}"

        # routing_divergence: pct of subtasks where LP and LLM chose different models
        # when_differed_lp_better: when they differed, how often did LP's choice get higher subtask score?
        total_subtasks = 0
        differed_count = 0
        when_differed_lp_better = 0
        when_differed_llm_better = 0
        when_differed_tie = 0
        for jid, mode_assigns in job_assignments.items():
            lp_a = mode_assigns.get("lp") or {}
            llm_a = mode_assigns.get("llm") or {}
            lp_scores = (job_subtask_scores.get(jid) or {}).get("lp") or {}
            llm_scores = (job_subtask_scores.get(jid) or {}).get("llm") or {}
            for sid in set(lp_a) | set(llm_a):
                lp_model = lp_a.get(sid)
                llm_model = llm_a.get(sid)
                if lp_model is None or llm_model is None:
                    continue
                total_subtasks += 1
                if lp_model != llm_model:
                    differed_count += 1
                    lp_sc = lp_scores.get(sid)
                    llm_sc = llm_scores.get(sid)
                    if lp_sc is not None and llm_sc is not None:
                        if lp_sc > llm_sc:
                            when_differed_lp_better += 1
                        elif llm_sc > lp_sc:
                            when_differed_llm_better += 1
                        else:
                            when_differed_tie += 1
        comparison["routing_divergence_pct"] = round(100.0 * differed_count / total_subtasks, 1) if total_subtasks else None
        comparison["when_differed_lp_better"] = when_differed_lp_better
        comparison["when_differed_llm_better"] = when_differed_llm_better
        comparison["when_differed_tie"] = when_differed_tie
        comparison["total_subtasks_compared"] = total_subtasks

        report["comparison_lp_vs_llm"] = comparison

    # Pareto frontier aggregates and data
    bench_config: dict[str, Any] = {}
    config_path = bench_dir / "bench_config.json"
    if config_path.exists():
        try:
            bench_config = _read_json(config_path)
        except Exception:
            pass

    lambda_token = float(bench_config.get("lambda_token", 0.5))
    lambda_switch = float(bench_config.get("lambda_switch", 0.2))
    horizon_depth = int(bench_config.get("horizon_depth", 1))
    quality_estimator_type = str(bench_config.get("quality_estimator_type", "static"))

    report["aggregates"] = {}
    report["pareto_data"] = {}
    report["pareto_frontier_indices"] = {}

    for bench_name in set(e.benchmark for e in examples):
        report["aggregates"][bench_name] = {}
        pareto_points: list[dict[str, Any]] = []
        for mode, mode_data in report["modes"].items():
            by_bench = mode_data.get("by_benchmark") or {}
            mean_score = by_bench.get(bench_name) if by_bench else mode_data.get("avg_score")
            mean_token = None
            mean_switch = None
            mean_vram = mode_data.get("avg_vram_used_gb")
            score_std = mode_data.get("score_std")
            # Get token/switch from mode-level or compute from rows
            bench_rows = [r for r in rows if r.get("benchmark") == bench_name and r.get("routing_mode") == mode]
            if bench_rows:
                tok_vals = [r.get("sum_token_cost") for r in bench_rows if isinstance(r.get("sum_token_cost"), (int, float))]
                switch_vals = [r.get("estimated_switch_cost_ms") for r in bench_rows if isinstance(r.get("estimated_switch_cost_ms"), (int, float))]
                mean_token = sum(tok_vals) / len(tok_vals) if tok_vals else None
                mean_switch = sum(switch_vals) / len(switch_vals) if switch_vals else None
            agg = {
                "mean_score": mean_score,
                "mean_token_cost": mean_token,
                "mean_switch_cost_est": mean_switch,
                "mean_peak_vram_used_gb": mean_vram,
                "score_std": score_std,
            }
            report["aggregates"][bench_name][mode] = agg
            pt = {
                "routing_mode": mode,
                "lambda_token": lambda_token,
                "lambda_switch": lambda_switch,
                "horizon_depth": horizon_depth,
                "quality_estimator_type": quality_estimator_type,
                "mean_score": mean_score,
                "mean_token_cost": mean_token,
                "mean_switch_cost_est": mean_switch,
                "mean_peak_vram_used_gb": mean_vram,
            }
            if pt["mean_score"] is not None:
                pareto_points.append(pt)
        report["pareto_data"][bench_name] = pareto_points

        # Pareto frontier: non-dominated (score vs cost). Sort by token_cost asc, keep max score so far.
        if pareto_points:
            by_token = sorted(
                [(i, p) for i, p in enumerate(pareto_points) if p.get("mean_token_cost") is not None],
                key=lambda x: (x[1].get("mean_token_cost") or float("inf"), -(x[1].get("mean_score") or 0)),
            )
            frontier: list[int] = []
            best_score = -float("inf")
            for i, p in by_token:
                s = p.get("mean_score") or 0
                if s >= best_score:
                    best_score = s
                    if i not in frontier:
                        frontier.append(i)
            report["pareto_frontier_indices"][bench_name] = frontier

    # Pareto CSV (append columns to existing rows)
    pareto_csv_path = bench_dir / "report_pareto.csv"
    if rows and report.get("pareto_data"):
        pareto_fieldnames = [
            "benchmark_name", "routing_mode", "lambda_token", "lambda_switch", "horizon_depth",
            "quality_estimator_type", "mean_score", "mean_token_cost", "mean_switch_cost_est", "mean_peak_vram_used_gb",
        ]
        pareto_rows: list[dict[str, Any]] = []
        for bench_name, points in report["pareto_data"].items():
            for p in points:
                pareto_rows.append({
                    "benchmark_name": bench_name,
                    "routing_mode": p.get("routing_mode", ""),
                    "lambda_token": p.get("lambda_token"),
                    "lambda_switch": p.get("lambda_switch"),
                    "horizon_depth": p.get("horizon_depth"),
                    "quality_estimator_type": p.get("quality_estimator_type"),
                    "mean_score": p.get("mean_score"),
                    "mean_token_cost": p.get("mean_token_cost"),
                    "mean_switch_cost_est": p.get("mean_switch_cost_est"),
                    "mean_peak_vram_used_gb": p.get("mean_peak_vram_used_gb"),
                })
        if pareto_rows:
            with pareto_csv_path.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=pareto_fieldnames)
                w.writeheader()
                w.writerows(pareto_rows)
            report["report_pareto_csv"] = str(pareto_csv_path)

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

