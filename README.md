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

## Benchmarks

Two built-in benchmarks (deterministic, small):

- `workflowbench`: multi-subtask synthetic workflows
- `code_math_mix`: mixed code+math+reasoning+extraction tasks

Run both routing modes and write `report.json` + `report.csv`:

```bash
swarm bench --benchmark workflowbench --compare both --limit 3
swarm bench --benchmark code_math_mix --compare both --limit 8
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

## Outputs

- Telemetry JSONL: `runs/<run_id>/telemetry.jsonl`
- Human-readable summary: `runs/<run_id>/summary.json`

## Notes

- VRAM footprints and load times are **hard-coded constants** in `src/swarm/agents.py` for MVP.
- Switching cost is approximated as a per-job penalty for activating models not already loaded.

