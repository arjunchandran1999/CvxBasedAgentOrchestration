# Data analysis utilities

This folder contains small scripts to analyze experiment outputs produced by:

- `swarm bench` (writes `bench_config.json`, `report.json`, `report.csv`, `outputs.jsonl`, and per-mode `runs/*/telemetry.jsonl`)
- `swarm experiment` (grid sweep; writes `index.json` that links to many `report.json`)
- `python scripts/run_full_sweep.py` (writes `runs/<timestamp>_full_sweep/**/report.json` and `meta_report.json`)

## Analyze the latest experiment batch

Run from the repo root:

```bash
python3 data_analysis/analyze_latest.py
```

Outputs are written under `data_analysis/out/<timestamp>/`:

- `analysis.md` (narrative: what happened, strengths/weaknesses, next experiments)
- `pareto.svg` (score vs token cost, with Pareto frontier)
- `switch_cost.svg` (score vs estimated switch cost)
- `vram.svg` (score vs peak VRAM used)

You can also point it at a specific `report.json` / `index.json` / `meta_report.json`:

```bash
python3 data_analysis/analyze_latest.py --input experiments/exp2_vram_48/20260221-235849-12139/report.json
```

