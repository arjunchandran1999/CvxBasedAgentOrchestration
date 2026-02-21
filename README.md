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

- \(x_{im} \in [0,1]\) — assignment: agent \(i\) handles subtask \(m\)
- \(y_i \in [0,1]\) — activation: agent \(i\) is loaded

**Parameters**

- \(P_{im}\) — estimated quality (performance) of agent \(i\) on subtask \(m\), in \([0,1]\)
- \(C^{\text{tok}}_{im}\) — normalized token cost = `(cost_per_token × estimated_tokens) / token_scale`
- \(C^{\text{sw}}_i\) — switch cost for agent \(i\): 0 if already loaded, else `load_time_ms / (switch_t_scale_ms × 3)`
- \(v_i\) — VRAM (GB) for agent \(i\)
- \(G\) — GPU VRAM capacity (GB)

**Objective** (maximize):

\[
\sum_{i,m} x_{im}\bigl(P_{im} - \lambda_{\text{token}}\, C^{\text{tok}}_{im}\bigr)
\;-\; \lambda_{\text{switch}} \sum_i C^{\text{sw}}_i\, y_i
\]

**Constraints**

- \(\sum_i x_{im} = 1\) for each \(m\) — each subtask assigned to exactly one agent
- \(x_{im} \leq y_i\) for all \(i,m\) — can assign to agent only if it is active
- \(\sum_i v_i\, y_i \leq G\) — VRAM budget

The LP chooses which models to load and how to assign subtasks so that quality minus token and switch costs is maximized, subject to VRAM.

## Benchmarks

Two built-in benchmarks (deterministic, small):

- `workflowbench`: multi-subtask synthetic workflows
- `code_math_mix`: mixed code+math+reasoning+extraction tasks

Run both routing modes and write `report.json` + `report.csv`:

```bash
swarm bench --benchmark workflowbench --compare both --limit 3
swarm bench --benchmark code_math_mix --compare both --limit 8
```

## Experiments (LP vs LLM sweeps)

**Experiment 1 — VRAM stress** (agents_heavy, 8GB, workflowbench):

```bash
swarm bench --benchmark workflowbench --agents_file configs/agents_heavy.json \
  --gpu_vram_gb 8 --compare both --limit 3 \
  --output_dir experiments/exp1_vram_stress
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

## Why LP may underperform LLM in some experiments

LP routing optimizes **estimated** performance from capability priors (agent.capabilities per task type), not actual observed performance. In Exp 1 (agents_heavy, 8GB):

- **LP** chose `qwen2.5-coder:7b` (code specialist, highest P−λC under 8GB). Qwen may over-explain math or extraction and produce verbose outputs that fail strict scorers.
- **LLM** chose `gemma2:2b` (smaller, cheaper). Gemma2 often generalizes better on simple extraction/math/summarize tasks and follows "output only the number" style.

Other factors:

- **8GB VRAM** allows only 1 heavy model. Both LP and LLM must pick a single model; no multi-model specialist routing is possible.
- **Capability priors** may be mis-calibrated (e.g. qwen code=0.85 but extraction/math may not reflect real behavior).
- **LLM planner** can sometimes approximate task-fit heuristically in ways the LP priors miss.

To see LP advantage, run with: higher VRAM (24GB), heterogeneous tasks where specialists clearly outperform, or higher λ_switch where LP’s explicit switch-cost optimization should reduce model swaps.

## Outputs

- **Telemetry**: `runs/<run_id>/telemetry.jsonl` — job, subtask, orchestration, and `subtask_plan` events
- **Summary**: `runs/<run_id>/summary.json`
- **Subtask plan**: `event="subtask_plan"` — the plan fed to LP or LLM before routing (`plan_source`: `benchmark` or `decomposer`)
- **Subtask outputs**: each subtask event includes `output`; `bench_dir/outputs.jsonl` after `swarm bench` has per-subtask outputs with expected/score
- **Multi-agent**: `n_distinct_models`, `n_role_agents`, `active_role_agents` in job telemetry

## Notes

- VRAM footprints and load times are **hard-coded constants** in agent JSON for MVP.
- Switching cost is approximated as a per-job penalty for activating models not already loaded.

