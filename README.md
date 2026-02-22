# Ollama Swarm Router (GPU-aware)

End-to-end MVP to compare exactly two routing modes:

- **`lp`**: CVXPY LP routing with token + VRAM + model-switching penalty.
- **`llm`**: planner LLM chooses routing from the same information (no CVXPY).

Telemetry is written as JSONL so you can quantify quality/cost/switching differences.

## Quickstart (uv)

```bash
cd /home/adevc/cvxpyHackathon
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Prereqs

- Ollama running locally (default base URL `http://localhost:11434`)
- Models pulled (names must match your local tags):
  - `qwen2.5-coder:7b`
  - `llama3.1:8b`
  - `phi4:14b`
  - `gemma2:2b`

## Run

Routing = LP:

```bash
swarm run \
  --routing lp \
  --gpu_vram_gb 24 \
  --lambda_token 0.5 \
  --lambda_switch 0.2 \
  --query "Write a Python function that parses a CSV and returns stats."
```

Routing = LLM planner:

```bash
swarm run \
  --routing llm \
  --gpu_vram_gb 24 \
  --query "Solve: if f(x)=x^2+3x+2, find its roots and explain."
```

Dry-run (no Ollama calls; useful for smoke testing end-to-end):

```bash
swarm run --dry_run --routing lp --query "Test job"
```

Batch run (so `loaded_models` carries across jobs):

```bash
swarm run --routing lp \
  --query "Write a Python function that parses a CSV." \
  --query "Now solve: integrate x^2 from 0 to 1." \
  --query "Summarize the previous two answers."
```

## Lambda parameters and optimization

LP routing solves a linear program to assign each subtask to an agent (model). Two trade-off parameters control the objective:

| Parameter | Default | Effect |
|-----------|---------|--------|
| **λ_token** | 0.5 | Weight on token cost. Higher → prefer cheaper / shorter outputs; lower → favor estimated quality more. |
| **λ_switch** | 0.2 | Weight on model switch cost. Higher → prefer reusing already-loaded models; lower → allow more model swaps for better fit. |

### Mathematical formulation

**Variables**

- $x_{im} \in [0,1]$ — assignment: agent $i$ handles subtask $m$
- $y_i \in [0,1]$ — activation: agent $i$ is loaded

**Parameters**

- $P_{im}$ — estimated quality (performance) of agent $i$ on subtask $m$, in $[0,1]$
- $C^{\text{tok}}_{im}$ — normalized token cost = `(cost_per_token × estimated_tokens) / token_scale`
- $C^{\text{sw}}_i$ — switch cost for agent $i$: 0 if already loaded, else `load_time_ms / (switch_t_scale_ms × 3)`
- $v_i$ — VRAM (GB) for agent $i$
- $G$ — GPU VRAM capacity (GB)

**Objective** (maximize):

$$
\sum_{i,m} x_{im}\bigl(P_{im} - \lambda_{\text{token}}\, C^{\text{tok}}_{im}\bigr)
\;-\; \lambda_{\text{switch}} \sum_i C^{\text{sw}}_i\, y_i
$$

**Constraints**

- $\sum_i x_{im} = 1$ for each $m$ — each subtask assigned to exactly one agent
- $x_{im} \leq y_i$ for all $i,m$ — can assign to agent only if it is active
- $\sum_i v_i\, y_i \leq G$ — VRAM budget

The LP chooses which models to load and how to assign subtasks so that quality minus token and switch costs is maximized, subject to VRAM.

The LP uses **HiGHS** by default, with automatic fallback to SCS on solver failure. Telemetry records `lp_solver_name`, `lp_solve_status`, and `lp_solve_time_ms` per plan step.

## Task DAGs and MPC

For benchmarks that define dependencies between subtasks (e.g. workflowbench), jobs run as **directed acyclic graphs (DAGs)**:

- **Pure DAG** (`--horizon_depth 0`): Route and execute layer-by-layer. At each layer, only currently ready nodes (deps satisfied) are optimized and run.

- **MPC** (`--horizon_depth 1` or higher): *Receding-horizon planning*. At each step, the LP/LLM optimizes over a **horizon** of ready nodes plus their successors (up to `horizon_depth` steps ahead), then **executes only the currently ready subset**. This lets the optimizer anticipate future tasks (e.g. pre-load models) while still replanning as the DAG unfolds.

```bash
# MPC with 1-layer lookahead (default for DAG benchmarks)
swarm bench --benchmark workflowbench --horizon_depth 1 --compare both --limit 3

# Pure layer-by-layer DAG (no lookahead)
swarm bench --benchmark workflowbench --horizon_depth 0 --compare both --limit 3
```

Telemetry events for DAG/MPC runs: `dag_structure` (nodes, edges, mpc, horizon_depth), `plan_step` (horizon_nodes, ready_now_nodes, lp_solver_name, executed_assignments), and `job_dag_summary` (n_plan_steps, n_layers, total costs).

## Benchmarks

Two built-in benchmarks (deterministic, small):

- **workflowbench**: multi-subtask synthetic workflows with **DAG dependencies** (extraction → math → code; s0→s1→s2). Uses MPC by default. Optional `--code_eval` for code execution.
- **code_math_mix**: mixed code+math+reasoning+extraction tasks (flat subtask lists)

Scoring uses per-subtask `expected` values (JSON match, numeric match, code execution when `--code_eval`). Artifacts store `model_task_estimates` — the utility matrix (quality, token_cost, switch_cost, utility per agent per subtask) — for analysis.

Run both routing modes and write `report.json` + `report.csv`:

```bash
swarm bench --benchmark workflowbench --compare both --limit 3
swarm bench --benchmark code_math_mix --compare both --limit 8
```

## Experiments (LP vs LLM sweeps)

See [experiments/PRESENTATION.md](experiments/PRESENTATION.md) for detailed analysis of exp1 results, LP pros/cons, and presentation material.

**Experiment 1 — VRAM stress** (agents_heavy, 8GB, workflowbench with MPC):

```bash
swarm bench --benchmark workflowbench --agents_file configs/agents_heavy.json \
  --gpu_vram_gb 8 --compare both --limit 3 \
  --horizon_depth 1 --output_dir experiments/exp1_vram_stress
```

**Experiment 2 — λ_switch sweep** (agents_fast, code_math_mix, interleave):

```bash
swarm experiment --suite code_math_sweep \
  --lambda_switches 0.0,0.2,0.5,1.0 --gpu_vrams 8 \
  --agents_file configs/agents_fast.json --limit 8 \
  --output_dir experiments/exp2_switch_sweep
```

**Experiment 3 — Interleave vs grouped** (run both 3a and 3b):

```bash
# 3a: grouped
swarm bench --benchmark code_math_mix --agents_file configs/agents_fast.json \
  --gpu_vram_gb 8 --lambda_switch 0.5 --compare both --limit 8 \
  --mix grouped --output_dir experiments/exp3_interleave_vs_grouped/grouped

# 3b: interleave
swarm bench --benchmark code_math_mix --agents_file configs/agents_fast.json \
  --gpu_vram_gb 8 --lambda_switch 0.5 --compare both --limit 8 \
  --mix interleave --output_dir experiments/exp3_interleave_vs_grouped/interleave
```

Or run all three via the script:

```bash
./scripts/run_experiments.sh
```

### Full sweep (timestamped, all benchmarks + Pareto)

Runs workflowbench, code_math_mix, and agentic_bench across VRAM/horizon/lambda configs into `runs/<timestamp>_full_sweep/` and writes `meta_report.json` aggregating Pareto metrics:

```bash
python scripts/run_full_sweep.py
```

Creates:
- `runs/<timestamp>_full_sweep/workflowbench/`, `code_math_mix/`, `agentic_bench/` subdirs
- `meta_report.json` — all `report.json` entries with `mean_score`, `mean_token_cost`, `mean_switch_cost_est`, `mean_peak_vram_used_gb` per (benchmark, routing_mode)

## Agent presets (fast POC vs heavy)

Agent registries are JSON files passed via `--agents_file`. This makes it easy to move
from a fast/tiny POC setup to a heavier compute environment without code changes.

- **Tiny/fast preset**: `configs/agents_fast.json` (e.g. `tinyllama:latest`, `gemma2:2b`, `phi3:mini`)
- **Heavy preset**: `configs/agents_heavy.json` (e.g. `qwen2.5-coder:7b`, `llama3.1:8b`, `phi4:14b`, `gemma2:2b`)

You can also generate a fast preset from whatever is already installed in your local Ollama:

```bash
swarm write-agents --preset auto_fast --out configs/agents_auto_fast_local.json --k 3
```

Then run with it:

```bash
swarm bench --benchmark workflowbench --compare both --limit 3 --agents_file configs/agents_auto_fast_local.json
```

Other factors:

- **8GB VRAM** allows only 1 heavy model. Both LP and LLM must pick a single model; no multi-model specialist routing is possible.
- **Capability priors** may be mis-calibrated (e.g. qwen code=0.85 but extraction/math may not reflect real behavior).
- **LLM planner** can sometimes approximate task-fit heuristically in ways the LP priors miss.

To see LP advantage, run with: higher VRAM (24GB), heterogeneous tasks where specialists clearly outperform, or higher λ_switch where LP’s explicit switch-cost optimization should reduce model swaps.

## Evaluation options

- **`--judge`** — Use an LLM to score each subtask output (0–1) with a rationale. Adds extra Ollama calls. Useful when benchmarks lack strict automated scorers.
- **`--code_eval`** — For code subtasks (workflowbench, humaneval, mbpp, code_math_mix), run extracted Python in a subprocess against expected outputs. **Executes untrusted code**; use only in trusted environments.

```bash
swarm bench --benchmark workflowbench --compare both --limit 3 --code_eval
swarm bench --benchmark code_math_mix --compare both --limit 8 --judge --code_eval
```

## Outputs

- **Telemetry**: `runs/<run_id>/telemetry.jsonl` — job, subtask, orchestration, and `subtask_plan` events
- **Summary**: `runs/<run_id>/summary.json` — per-job summary with `plan`, `task_to_model`, `job_score`, `success_rate`, `models_swapped_in`, `vram_used_gb`, etc.
- **Artifacts**: `runs/<run_id>/artifacts/<job_id>.json` — full job record: `benchmark_reference` (subtasks with `expected`), `subtasks`, `assignments`, `model_task_estimates` (quality, token_cost, switch_cost, utility per model per subtask), `results` (subtask outputs)
- **Subtask plan**: `event="subtask_plan"` — the plan fed to LP or LLM before routing (`plan_source`: `benchmark` or `decomposer`)
- **Subtask outputs**: each subtask event includes `output`; `bench_dir/outputs.jsonl` after `swarm bench` has per-subtask outputs with `expected`, `benchmark_score`, `agent`
- **Report**: `bench_dir/report.json` — aggregated scores by mode (lp/llm): `avg_score`, `avg_models_swapped_in`, `avg_estimated_switch_cost_ms`, `avg_active_model_count`, `avg_vram_used_gb`, `score_std`, `vram_violation_rate`, `avg_oracle_gap_pct` (LLM), `avg_lp_objective_value` (LP); `comparison_lp_vs_llm` with `jobs_lp_won`, `jobs_llm_won`, `routing_divergence_pct`, `when_differed_lp_better`; `bench_dir/report.csv` for per-example rows
- **Pareto**: `report.json` also has `aggregates`, `pareto_data`, `pareto_frontier_indices` (score vs token/switch/VRAM); `report_pareto.csv` for plotting
- **DAG/MPC**: `dag_structure`, `plan_step`, `job_dag_summary` events in telemetry when running workflowbench or agentic_bench (or any benchmark with `get_task_dag`)
- **Multi-agent**: `n_distinct_models`, `n_role_agents`, `active_role_agents` in job telemetry

## Notes

- VRAM footprints and load times are **hard-coded constants** in agent JSON for MVP.
- Switching cost is approximated as a per-job penalty for activating models not already loaded.

