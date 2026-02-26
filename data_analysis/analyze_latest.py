#!/usr/bin/env python3
"""
Analyze the latest experiment outputs and generate visualizations.

This script is dependency-light (stdlib only). It reads the repo's generated artifacts:
  - report.json (from swarm bench)
  - index.json (from swarm experiment)
  - meta_report.json (from scripts/run_full_sweep.py)

It writes an analysis bundle under data_analysis/out/<timestamp>/ including:
  - analysis.md
  - pareto.svg (score vs mean token cost, with Pareto frontier)
  - switch_cost.svg (score vs mean estimated switch cost)
  - vram.svg (score vs mean peak VRAM used)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional


REPO_ROOT = Path(__file__).resolve().parent.parent


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _now_tag() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def _safe_float(x: Any) -> float | None:
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    return None


def _safe_str(x: Any) -> str | None:
    if x is None:
        return None
    if isinstance(x, str):
        return x
    return str(x)


def _iter_files(root: Path, name: str) -> Iterable[Path]:
    if not root.exists():
        return []
    return root.rglob(name)


def _pick_latest(paths: Iterable[Path]) -> Path | None:
    best: tuple[float, Path] | None = None
    for p in paths:
        try:
            ts = p.stat().st_mtime
        except OSError:
            continue
        if best is None or ts > best[0]:
            best = (ts, p)
    return best[1] if best else None


@dataclass(frozen=True)
class ParetoPoint:
    benchmark_name: str
    routing_mode: str  # "lp" | "llm"
    mean_score: float
    mean_token_cost: float | None
    mean_switch_cost_est: float | None
    mean_peak_vram_used_gb: float | None
    lambda_token: float | None = None
    lambda_switch: float | None = None
    horizon_depth: int | None = None
    gpu_vram_gb: float | None = None
    bench_dir: str | None = None
    source: str | None = None  # report path / meta_report path


@dataclass(frozen=True)
class MultiModelMetrics:
    bench_dir: str
    benchmark_name: str
    routing_mode: str  # "lp" | "llm"
    n_jobs: int
    mean_distinct_assigned_models: float | None
    frac_jobs_multi_assigned: float | None
    mean_active_models: float | None
    mean_models_swapped_in: float | None


def _points_from_report(report_path: Path) -> tuple[list[ParetoPoint], dict[str, Any]]:
    """
    Returns (pareto_points, context).
    context includes a minimal set of fields used for narrative.
    """
    rep = _read_json(report_path)
    bench_dir = Path(rep.get("bench_dir") or report_path.parent)
    if not bench_dir.is_absolute():
        bench_dir = (REPO_ROOT / bench_dir).resolve()

    bench_cfg_path = bench_dir / "bench_config.json"
    bench_cfg: dict[str, Any] = _read_json(bench_cfg_path) if bench_cfg_path.exists() else {}
    gpu_vram = _safe_float(bench_cfg.get("gpu_vram_gb"))

    pts: list[ParetoPoint] = []
    pareto_data = rep.get("pareto_data") or {}
    for bench_name, items in pareto_data.items():
        if not isinstance(items, list):
            continue
        for it in items:
            if not isinstance(it, dict):
                continue
            s = _safe_float(it.get("mean_score"))
            if s is None:
                continue
            pts.append(
                ParetoPoint(
                    benchmark_name=str(bench_name),
                    routing_mode=str(it.get("routing_mode") or ""),
                    mean_score=s,
                    mean_token_cost=_safe_float(it.get("mean_token_cost")),
                    mean_switch_cost_est=_safe_float(it.get("mean_switch_cost_est")),
                    mean_peak_vram_used_gb=_safe_float(it.get("mean_peak_vram_used_gb")),
                    lambda_token=_safe_float(it.get("lambda_token")),
                    lambda_switch=_safe_float(it.get("lambda_switch")),
                    horizon_depth=int(it["horizon_depth"]) if isinstance(it.get("horizon_depth"), int) else None,
                    gpu_vram_gb=gpu_vram,
                    bench_dir=str(bench_dir),
                    source=str(report_path),
                )
            )

    ctx = {
        "report_path": str(report_path),
        "bench_dir": str(bench_dir),
        "bench_config": bench_cfg,
        "modes": rep.get("modes") or {},
        "comparison_lp_vs_llm": rep.get("comparison_lp_vs_llm") or {},
    }
    return pts, ctx


def _points_from_index(index_path: Path) -> tuple[list[ParetoPoint], dict[str, Any]]:
    idx = _read_json(index_path)
    runs = idx.get("runs") or []
    pts: list[ParetoPoint] = []
    ctxs: list[dict[str, Any]] = []
    for r in runs:
        if not isinstance(r, dict):
            continue
        rp = r.get("report")
        if not rp:
            continue
        report_path = Path(str(rp))
        if not report_path.is_absolute():
            report_path = (REPO_ROOT / report_path).resolve()
        if not report_path.exists():
            continue
        p, c = _points_from_report(report_path)
        pts.extend(p)
        ctxs.append(c)
    return pts, {"index_path": str(index_path), "reports": ctxs}


def _points_from_meta_report(meta_path: Path) -> tuple[list[ParetoPoint], dict[str, Any]]:
    meta = _read_json(meta_path)
    entries = meta.get("reports") or []
    pts: list[ParetoPoint] = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        s = _safe_float(e.get("mean_score"))
        if s is None:
            continue
        pts.append(
            ParetoPoint(
                benchmark_name=str(e.get("benchmark_name") or "unknown"),
                routing_mode=str(e.get("routing_mode") or ""),
                mean_score=s,
                mean_token_cost=_safe_float(e.get("mean_token_cost")),
                mean_switch_cost_est=_safe_float(e.get("mean_switch_cost_est")),
                mean_peak_vram_used_gb=_safe_float(e.get("mean_peak_vram_used_gb")),
                lambda_token=_safe_float(e.get("lambda_token")),
                lambda_switch=_safe_float(e.get("lambda_switch")),
                horizon_depth=int(e["horizon_depth"]) if isinstance(e.get("horizon_depth"), int) else None,
                gpu_vram_gb=_safe_float(e.get("gpu_vram_gb")),
                bench_dir=_safe_str(e.get("bench_dir")),
                source=str(meta_path),
            )
        )
    return pts, {"meta_report_path": str(meta_path), "n_entries": len(pts)}


def _detect_latest_input() -> Path | None:
    # Prefer full-sweep meta_report.json (covers many runs).
    meta = _pick_latest(_iter_files(REPO_ROOT / "runs", "meta_report.json"))
    if meta:
        return meta
    # Next, prefer swarm experiment index.json (grid sweep).
    idx = _pick_latest(_iter_files(REPO_ROOT / "experiments", "index.json"))
    if idx:
        return idx
    # Fall back to the latest report.json anywhere under experiments/runs/bench_runs*.
    candidates: list[Path] = []
    for root in [REPO_ROOT / "experiments", REPO_ROOT / "runs", REPO_ROOT / "bench_runs"]:
        candidates.extend(list(_iter_files(root, "report.json")))
    return _pick_latest(candidates)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        if isinstance(d, dict):
            out.append(d)
    return out


def _compute_multi_model_metrics_for_bench_dir(bench_dir: Path) -> list[MultiModelMetrics]:
    """
    Deeper than report aggregates: parses `telemetry.jsonl` to compute distinct assigned models/job.
    """
    report_path = bench_dir / "report.json"
    cfg_path = bench_dir / "bench_config.json"
    if not report_path.exists():
        return []
    rep = _read_json(report_path)
    cfg = _read_json(cfg_path) if cfg_path.exists() else {}
    benchmark_name = str((cfg.get("benchmarks") or ["unknown"])[0])

    out: list[MultiModelMetrics] = []
    modes = rep.get("modes") or {}
    for mode, md in modes.items():
        run_dir = md.get("run_dir")
        if not run_dir:
            continue
        run_path = Path(str(run_dir))
        if not run_path.is_absolute():
            run_path = (REPO_ROOT / run_path).resolve()
        evs = _load_jsonl(run_path / "telemetry.jsonl")

        assigned: dict[str, set[str]] = {}
        active_counts: list[float] = []
        swapped_counts: list[float] = []

        for e in evs:
            if e.get("event") == "subtask":
                jid = str(e.get("job_id") or "")
                ag = str(e.get("assigned_agent") or "")
                if jid and ag:
                    assigned.setdefault(jid, set()).add(ag)
            if e.get("event") in {"job", "job_costs"}:
                am = e.get("active_models") or []
                if isinstance(am, list):
                    active_counts.append(float(len(am)))
                ms = e.get("models_swapped_in") or []
                if isinstance(ms, list):
                    swapped_counts.append(float(len(ms)))

        distinct_counts = [len(s) for s in assigned.values()]
        n_jobs = len(distinct_counts)
        mean_distinct = (sum(distinct_counts) / n_jobs) if n_jobs else None
        frac_multi = (sum(1 for c in distinct_counts if c > 1) / n_jobs) if n_jobs else None
        mean_active = (sum(active_counts) / len(active_counts)) if active_counts else None
        mean_swapped = (sum(swapped_counts) / len(swapped_counts)) if swapped_counts else None

        out.append(
            MultiModelMetrics(
                bench_dir=str(bench_dir),
                benchmark_name=benchmark_name,
                routing_mode=str(mode),
                n_jobs=n_jobs,
                mean_distinct_assigned_models=mean_distinct,
                frac_jobs_multi_assigned=frac_multi,
                mean_active_models=mean_active,
                mean_models_swapped_in=mean_swapped,
            )
        )

    return out


def _pareto_frontier(points: list[ParetoPoint]) -> list[int]:
    """
    2D frontier: minimize mean_token_cost, maximize mean_score.
    Points missing token_cost are excluded.
    Returns indices into points.
    """
    idx_pts = [(i, p) for i, p in enumerate(points) if p.mean_token_cost is not None]
    idx_pts.sort(key=lambda x: (x[1].mean_token_cost or float("inf"), -(x[1].mean_score or 0.0)))
    frontier: list[int] = []
    best = -float("inf")
    for i, p in idx_pts:
        if p.mean_score >= best:
            best = p.mean_score
            frontier.append(i)
    return frontier


def _svg_scatter(
    *,
    points: list[ParetoPoint],
    x_fn,
    y_fn,
    x_label: str,
    y_label: str,
    title: str,
    out_path: Path,
    frontier_indices: list[int] | None = None,
) -> None:
    # Filter to points with both x and y.
    xy = []
    for i, p in enumerate(points):
        x = x_fn(p)
        y = y_fn(p)
        if x is None or y is None:
            continue
        xy.append((i, float(x), float(y), p))
    if not xy:
        out_path.write_text(f"<svg xmlns='http://www.w3.org/2000/svg' width='800' height='120'><text x='10' y='40'>No data for plot: {title}</text></svg>\n", encoding="utf-8")
        return

    W, H = 900, 560
    m = 70
    plot_w, plot_h = W - 2 * m, H - 2 * m

    xs = [x for _, x, _, _ in xy]
    ys = [y for _, _, y, _ in xy]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if math.isclose(xmin, xmax):
        xmax = xmin + 1.0
    if math.isclose(ymin, ymax):
        ymax = ymin + 1.0

    def sx(x: float) -> float:
        return m + (x - xmin) / (xmax - xmin) * plot_w

    def sy(y: float) -> float:
        return m + plot_h - (y - ymin) / (ymax - ymin) * plot_h

    def color(mode: str) -> str:
        return {"lp": "#2563eb", "llm": "#f97316"}.get(mode, "#64748b")

    # Build frontier polyline if requested
    frontier_poly = ""
    if frontier_indices:
        fxy = []
        for i in frontier_indices:
            if i < 0 or i >= len(points):
                continue
            p = points[i]
            x = x_fn(p)
            y = y_fn(p)
            if x is None or y is None:
                continue
            fxy.append((float(x), float(y)))
        fxy.sort(key=lambda t: t[0])
        if len(fxy) >= 2:
            pts_str = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in fxy)
            frontier_poly = f"<polyline points='{pts_str}' fill='none' stroke='#111827' stroke-width='2' stroke-dasharray='6,5' opacity='0.8' />"

    # Axes & ticks (simple)
    tick_n = 5
    xticks = [xmin + (xmax - xmin) * k / tick_n for k in range(tick_n + 1)]
    yticks = [ymin + (ymax - ymin) * k / tick_n for k in range(tick_n + 1)]

    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' viewBox='0 0 {W} {H}'>",
        "<rect x='0' y='0' width='100%' height='100%' fill='white'/>",
        f"<text x='{m}' y='28' font-size='18' font-family='sans-serif' fill='#111827'>{_escape(title)}</text>",
        # Axes
        f"<line x1='{m}' y1='{m}' x2='{m}' y2='{m+plot_h}' stroke='#111827' stroke-width='1'/>",
        f"<line x1='{m}' y1='{m+plot_h}' x2='{m+plot_w}' y2='{m+plot_h}' stroke='#111827' stroke-width='1'/>",
        # Labels
        f"<text x='{m+plot_w/2:.1f}' y='{H-18}' font-size='14' font-family='sans-serif' fill='#111827' text-anchor='middle'>{_escape(x_label)}</text>",
        f"<text x='18' y='{m+plot_h/2:.1f}' font-size='14' font-family='sans-serif' fill='#111827' text-anchor='middle' transform='rotate(-90 18 {m+plot_h/2:.1f})'>{_escape(y_label)}</text>",
    ]

    # Ticks
    for t in xticks:
        x = sx(t)
        parts.append(f"<line x1='{x:.1f}' y1='{m+plot_h}' x2='{x:.1f}' y2='{m+plot_h+6}' stroke='#111827'/>")
        parts.append(f"<text x='{x:.1f}' y='{m+plot_h+24}' font-size='11' font-family='monospace' fill='#334155' text-anchor='middle'>{t:.2f}</text>")
    for t in yticks:
        y = sy(t)
        parts.append(f"<line x1='{m-6}' y1='{y:.1f}' x2='{m}' y2='{y:.1f}' stroke='#111827'/>")
        parts.append(f"<text x='{m-10}' y='{y+4:.1f}' font-size='11' font-family='monospace' fill='#334155' text-anchor='end'>{t:.2f}</text>")

    # Frontier overlay
    if frontier_poly:
        parts.append(frontier_poly)

    # Points
    for _, x, y, p in xy:
        cx, cy = sx(x), sy(y)
        c = color(p.routing_mode)
        tooltip = f"{p.routing_mode} | {p.benchmark_name} | score={p.mean_score:.3f}"
        if p.mean_token_cost is not None:
            tooltip += f" | tok={p.mean_token_cost:.1f}"
        if p.lambda_switch is not None:
            tooltip += f" | ls={p.lambda_switch}"
        if p.horizon_depth is not None:
            tooltip += f" | hd={p.horizon_depth}"
        parts.append(
            f"<circle cx='{cx:.1f}' cy='{cy:.1f}' r='6' fill='{c}' opacity='0.85'><title>{_escape(tooltip)}</title></circle>"
        )

    # Legend
    parts.append(f"<rect x='{W-m-170}' y='{m-10}' width='160' height='60' fill='white' stroke='#e2e8f0'/>")
    parts.append(f"<circle cx='{W-m-150}' cy='{m+10}' r='6' fill='#2563eb'/><text x='{W-m-135}' y='{m+14}' font-family='sans-serif' font-size='12'>LP</text>")
    parts.append(f"<circle cx='{W-m-150}' cy='{m+35}' r='6' fill='#f97316'/><text x='{W-m-135}' y='{m+39}' font-family='sans-serif' font-size='12'>LLM</text>")

    parts.append("</svg>\n")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def _escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\"", "&quot;")
        .replace("'", "&apos;")
    )


def _describe_points(points: list[ParetoPoint]) -> dict[str, Any]:
    by_mode: dict[str, list[ParetoPoint]] = {}
    for p in points:
        by_mode.setdefault(p.routing_mode or "unknown", []).append(p)

    def mean(vals: list[float]) -> float | None:
        if not vals:
            return None
        return sum(vals) / len(vals)

    out: dict[str, Any] = {"n_points": len(points), "by_mode": {}}
    for mode, xs in by_mode.items():
        out["by_mode"][mode] = {
            "n": len(xs),
            "mean_score": mean([p.mean_score for p in xs]),
            "mean_token_cost": mean([p.mean_token_cost for p in xs if p.mean_token_cost is not None]),
            "mean_switch_cost_est": mean([p.mean_switch_cost_est for p in xs if p.mean_switch_cost_est is not None]),
            "mean_peak_vram_used_gb": mean([p.mean_peak_vram_used_gb for p in xs if p.mean_peak_vram_used_gb is not None]),
        }
    return out


def _write_analysis_md(
    *,
    out_dir: Path,
    input_path: Path,
    points: list[ParetoPoint],
    ctx: dict[str, Any],
    frontier_indices: list[int],
) -> None:
    stats = _describe_points(points)
    frontier = [points[i] for i in frontier_indices if 0 <= i < len(points)]

    # Telemetry-derived multi-model usage (stronger than mode-level aggregates).
    mm_all: list[MultiModelMetrics] = []
    bench_dirs = sorted({p.bench_dir for p in points if p.bench_dir})
    for bd in bench_dirs:
        try:
            mm_all.extend(_compute_multi_model_metrics_for_bench_dir(Path(str(bd))))
        except Exception:
            continue

    # Identify best points by score and by efficiency (score/token).
    best_score = max(points, key=lambda p: p.mean_score, default=None)
    best_eff = None
    best_eff_val = -float("inf")
    for p in points:
        if p.mean_token_cost and p.mean_token_cost > 0:
            eff = p.mean_score / p.mean_token_cost
            if eff > best_eff_val:
                best_eff_val = eff
                best_eff = p

    # Try to explain "what happened and why" using mode-level aggregates if available.
    narrative_bits: list[str] = []
    comp = ctx.get("comparison_lp_vs_llm") or {}
    if comp:
        div = comp.get("routing_divergence_pct")
        if div is not None:
            narrative_bits.append(f"- Routing divergence: **{div}%** of subtasks got different model choices between LP and LLM.")
        for k, label in [
            ("avg_score_lp_minus_llm", "Avg score (LP − LLM)"),
            ("avg_total_latency_ms_lp_minus_llm", "Avg total latency ms (LP − LLM)"),
            ("avg_vram_used_gb_lp_minus_llm", "Avg VRAM used GB (LP − LLM)"),
            ("avg_estimated_switch_cost_ms_lp_minus_llm", "Avg estimated switch cost ms (LP − LLM)"),
        ]:
            if k in comp and comp[k] is not None:
                narrative_bits.append(f"- {label}: **{comp[k]}**")

    mode_aggs = stats.get("by_mode") or {}
    strengths: list[str] = []
    weaknesses: list[str] = []
    if "lp" in mode_aggs and "llm" in mode_aggs:
        lp = mode_aggs["lp"]
        llm = mode_aggs["llm"]
        # Heuristic comparisons
        if lp.get("mean_score") is not None and llm.get("mean_score") is not None:
            if lp["mean_score"] > llm["mean_score"] + 1e-6:
                strengths.append("LP achieves higher mean score than LLM on this batch.")
                weaknesses.append("LLM routing underperforms LP on mean score; may need better priors or planner prompts.")
            elif llm["mean_score"] > lp["mean_score"] + 1e-6:
                strengths.append("LLM achieves higher mean score than LP on this batch.")
                weaknesses.append("LP objective/priors may be mis-calibrated for these tasks.")
            else:
                strengths.append("LP and LLM reach similar mean score on this batch (tie-ish).")

        # Cost/efficiency
        if lp.get("mean_token_cost") and llm.get("mean_token_cost"):
            if lp["mean_token_cost"] < llm["mean_token_cost"]:
                strengths.append("LP is cheaper in token-cost on average (per report aggregates).")
            else:
                weaknesses.append("LP is more expensive in token-cost on average; consider tuning λ_token or token estimates.")
    else:
        weaknesses.append("Not enough cross-mode data to compare LP vs LLM (missing mode points).")

    # Next experiments (generic but concrete)
    next_exps = [
        "Increase `--limit` and run multiple `--seed` values to get confidence intervals (current runs are often N≤8).",
        "Run a **2D sweep** over `lambda_switch` × `horizon_depth` on DAG benchmarks (e.g. workflowbench) to quantify the lookahead benefit.",
        "Add a `lambda_token` sweep alongside `lambda_switch` to separate quality-vs-cost from switch-vs-reuse effects.",
        "Run with `--judge` on a subset and compare: static priors vs online-updated `QualityEstimator` (does routing improve over time?).",
        "Ablate the LP objective: (a) no switch penalty, (b) no token penalty, (c) no VRAM constraint, to validate each term’s contribution.",
        "Test sensitivity to agent sets (`agents_fast` vs `agents_heavy`) and to cost model assumptions (`cost_per_token`, `load_time_ms`).",
        "Add another benchmark (or expand workflowbench task templates) where routing choices are known to matter, then check if the same Pareto trend holds.",
    ]

    lines: list[str] = []
    lines.append(f"# Experiment analysis ({_now_tag()})")
    lines.append("")
    lines.append(f"**Input**: `{input_path}`")
    lines.append("")
    lines.append("## What was analyzed")
    lines.append("")
    lines.append(f"- **Pareto points**: {len(points)}")
    lines.append(f"- **Frontier points** (score vs token-cost): {len(frontier)}")
    lines.append("")
    lines.append("## Key visuals")
    lines.append("")
    lines.append("- `pareto.svg`: score vs mean token cost (frontier dashed)")
    lines.append("- `switch_cost.svg`: score vs mean estimated switch cost")
    lines.append("- `vram.svg`: score vs mean peak VRAM used")
    lines.append("")
    lines.append("## Summary stats (by routing mode)")
    lines.append("")
    for mode, d in (stats.get("by_mode") or {}).items():
        lines.append(f"- **{mode}**: n={d.get('n')}, mean_score={_fmt(d.get('mean_score'))}, mean_token_cost={_fmt(d.get('mean_token_cost'))}, mean_switch_cost={_fmt(d.get('mean_switch_cost_est'))}, mean_vram={_fmt(d.get('mean_peak_vram_used_gb'))}")
    lines.append("")
    lines.append("## What happened (and why this likely happened)")
    lines.append("")
    if narrative_bits:
        lines.extend(narrative_bits)
    else:
        lines.append("- Not enough `comparison_lp_vs_llm` info in the selected input to attribute causes; using Pareto trends only.")
    lines.append("")
    if best_score:
        lines.append("### Best observed score point")
        lines.append("")
        lines.append(_describe_point(best_score))
        lines.append("")
    if best_eff:
        lines.append("### Best observed score-per-token point")
        lines.append("")
        lines.append(_describe_point(best_eff))
        lines.append("")
    lines.append("## Strengths of the current results")
    lines.append("")
    for s in strengths or ["The pipeline produces consistent artifacts (reports, Pareto aggregates, and telemetry) that support systematic analysis."]:
        lines.append(f"- {s}")
    lines.append("")
    lines.append("## Weaknesses / gaps (what prevents strong conclusions)")
    lines.append("")
    for w in weaknesses or ["Small sample sizes and limited benchmark diversity make it hard to claim generality."]:
        lines.append(f"- {w}")
    lines.append("")
    lines.append("## Experiments to run next (to back the method more conclusively)")
    lines.append("")
    for e in next_exps:
        lines.append(f"- {e}")
    lines.append("")

    lines.append("## Deep dive: multi-agent usage (from telemetry)")
    lines.append("")
    if not mm_all:
        lines.append("- Telemetry-derived multi-agent metrics unavailable (missing telemetry.jsonl or missing bench_dir pointers).")
        lines.append("")
    else:
        by: dict[tuple[str, str], list[MultiModelMetrics]] = {}
        for m in mm_all:
            by.setdefault((m.benchmark_name, m.routing_mode), []).append(m)

        def _mean(vals):
            vals = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
            return sum(vals) / len(vals) if vals else None

        for (bn, mode), xs in sorted(by.items()):
            lines.append(
                f"- **{bn} / {mode}**: runs={len(xs)}, "
                f"mean distinct assigned models/job={_fmt(_mean([x.mean_distinct_assigned_models for x in xs]))}, "
                f"frac jobs w/ >1 assigned model={_fmt(_mean([x.frac_jobs_multi_assigned for x in xs]))}, "
                f"mean active_models/job={_fmt(_mean([x.mean_active_models for x in xs]))}, "
                f"mean models_swapped_in/job={_fmt(_mean([x.mean_models_swapped_in for x in xs]))}"
            )
        lines.append("")

    (out_dir / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(x: Any) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.4g}"
    return str(x)


def _describe_point(p: ParetoPoint) -> str:
    bits = [
        f"- **benchmark**: {p.benchmark_name}",
        f"- **mode**: {p.routing_mode}",
        f"- **mean_score**: {p.mean_score:.4f}",
    ]
    if p.mean_token_cost is not None:
        bits.append(f"- **mean_token_cost**: {p.mean_token_cost:.2f}")
    if p.mean_switch_cost_est is not None:
        bits.append(f"- **mean_switch_cost_est**: {p.mean_switch_cost_est:.2f} ms")
    if p.mean_peak_vram_used_gb is not None:
        bits.append(f"- **mean_peak_vram_used_gb**: {p.mean_peak_vram_used_gb:.2f} GB")
    if p.lambda_token is not None:
        bits.append(f"- **lambda_token**: {p.lambda_token}")
    if p.lambda_switch is not None:
        bits.append(f"- **lambda_switch**: {p.lambda_switch}")
    if p.horizon_depth is not None:
        bits.append(f"- **horizon_depth**: {p.horizon_depth}")
    if p.gpu_vram_gb is not None:
        bits.append(f"- **gpu_vram_gb**: {p.gpu_vram_gb}")
    if p.bench_dir:
        bits.append(f"- **bench_dir**: `{p.bench_dir}`")
    return "\n".join(bits)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None, help="Path to report.json, index.json, or meta_report.json. If omitted, auto-detect latest.")
    ap.add_argument("--out", default=None, help="Output directory (default: data_analysis/out/<timestamp>).")
    args = ap.parse_args()

    input_path = Path(args.input).expanduser() if args.input else _detect_latest_input()
    if input_path is None:
        raise SystemExit("No experiment artifacts found under runs/ or experiments/.")
    if not input_path.is_absolute():
        input_path = (REPO_ROOT / input_path).resolve()
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    if input_path.name == "meta_report.json":
        points, ctx = _points_from_meta_report(input_path)
    elif input_path.name == "index.json":
        points, ctx = _points_from_index(input_path)
    elif input_path.name == "report.json":
        points, ctx = _points_from_report(input_path)
    else:
        # Guess by content
        data = _read_json(input_path)
        if isinstance(data, dict) and "reports" in data and "runs_root" in data:
            points, ctx = _points_from_meta_report(input_path)
        elif isinstance(data, dict) and "runs" in data:
            points, ctx = _points_from_index(input_path)
        else:
            points, ctx = _points_from_report(input_path)

    if not points:
        raise SystemExit(f"No Pareto points found in: {input_path}")

    out_dir = Path(args.out).expanduser() if args.out else (REPO_ROOT / "data_analysis" / "out" / _now_tag())
    out_dir.mkdir(parents=True, exist_ok=True)

    frontier_idx = _pareto_frontier(points)

    _svg_scatter(
        points=points,
        x_fn=lambda p: p.mean_token_cost,
        y_fn=lambda p: p.mean_score,
        x_label="Mean token cost (lower is better)",
        y_label="Mean score (higher is better)",
        title="Pareto: score vs token cost",
        out_path=out_dir / "pareto.svg",
        frontier_indices=frontier_idx,
    )
    _svg_scatter(
        points=points,
        x_fn=lambda p: p.mean_switch_cost_est,
        y_fn=lambda p: p.mean_score,
        x_label="Mean estimated switch cost (ms, lower is better)",
        y_label="Mean score (higher is better)",
        title="Score vs switch cost (est.)",
        out_path=out_dir / "switch_cost.svg",
    )
    _svg_scatter(
        points=points,
        x_fn=lambda p: p.mean_peak_vram_used_gb,
        y_fn=lambda p: p.mean_score,
        x_label="Mean peak VRAM used (GB, lower is better)",
        y_label="Mean score (higher is better)",
        title="Score vs peak VRAM used",
        out_path=out_dir / "vram.svg",
    )

    _write_analysis_md(out_dir=out_dir, input_path=input_path, points=points, ctx=ctx, frontier_indices=frontier_idx)

    print(f"Wrote analysis bundle to: {out_dir}")
    print(f"- {out_dir / 'analysis.md'}")
    print(f"- {out_dir / 'pareto.svg'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

