# Experiment analysis (20260225-221621)

**Input**: `/home/adevc/cvxpyHackathon/runs/20260226-000113_full_sweep/meta_report.json`

## What was analyzed

- **Pareto points**: 18
- **Frontier points** (score vs token-cost): 8

## Key visuals

- `pareto.svg`: score vs mean token cost (frontier dashed)
- `switch_cost.svg`: score vs mean estimated switch cost
- `vram.svg`: score vs mean peak VRAM used

## Summary stats (by routing mode)

- **lp**: n=9, mean_score=0.6615, mean_token_cost=362.2, mean_switch_cost=723.1, mean_vram=4.472
- **llm**: n=9, mean_score=0.5905, mean_token_cost=396.3, mean_switch_cost=948.6, mean_vram=4.741

## What happened (and why this likely happened)

- Not enough `comparison_lp_vs_llm` info in the selected input to attribute causes; using Pareto trends only.

### Best observed score point

- **benchmark**: code_math_mix
- **mode**: lp
- **mean_score**: 1.0000
- **mean_token_cost**: 59.06
- **mean_switch_cost_est**: 150.00 ms
- **mean_peak_vram_used_gb**: 2.00 GB
- **lambda_token**: 0.5
- **lambda_switch**: 0.5
- **horizon_depth**: 1
- **gpu_vram_gb**: 48.0
- **bench_dir**: `runs/20260226-000113_full_sweep/code_math_mix/mix_interleave/20260226-000530-34426`

### Best observed score-per-token point

- **benchmark**: code_math_mix
- **mode**: lp
- **mean_score**: 1.0000
- **mean_token_cost**: 59.06
- **mean_switch_cost_est**: 150.00 ms
- **mean_peak_vram_used_gb**: 2.00 GB
- **lambda_token**: 0.5
- **lambda_switch**: 0.5
- **horizon_depth**: 1
- **gpu_vram_gb**: 48.0
- **bench_dir**: `runs/20260226-000113_full_sweep/code_math_mix/mix_interleave/20260226-000530-34426`

## Strengths of the current results

- LP achieves higher mean score than LLM on this batch.
- LP is cheaper in token-cost on average (per report aggregates).

## Weaknesses / gaps (what prevents strong conclusions)

- LLM routing underperforms LP on mean score; may need better priors or planner prompts.

## Experiments to run next (to back the method more conclusively)

- Increase `--limit` and run multiple `--seed` values to get confidence intervals (current runs are often N≤8).
- Run a **2D sweep** over `lambda_switch` × `horizon_depth` on DAG benchmarks (e.g. workflowbench) to quantify the lookahead benefit.
- Add a `lambda_token` sweep alongside `lambda_switch` to separate quality-vs-cost from switch-vs-reuse effects.
- Run with `--judge` on a subset and compare: static priors vs online-updated `QualityEstimator` (does routing improve over time?).
- Ablate the LP objective: (a) no switch penalty, (b) no token penalty, (c) no VRAM constraint, to validate each term’s contribution.
- Test sensitivity to agent sets (`agents_fast` vs `agents_heavy`) and to cost model assumptions (`cost_per_token`, `load_time_ms`).
- Add another benchmark (or expand workflowbench task templates) where routing choices are known to matter, then check if the same Pareto trend holds.

