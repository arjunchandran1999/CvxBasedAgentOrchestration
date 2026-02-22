#!/usr/bin/env bash
# Run experiments: LP vs LLM Planner Routing
# Execute from project root: ./scripts/run_experiments.sh
#
# Full sweep (all benchmarks, timestamped): python scripts/run_full_sweep.py
# Creates runs/<timestamp>_full_sweep/ with meta_report.json (Pareto metrics).

set -e
cd "$(dirname "$0")/.."
mkdir -p experiments runs

echo "=== Exp 1: VRAM stress (agents_heavy, gpu=8) ==="
swarm bench --benchmark workflowbench --agents_file configs/agents_heavy.json \
  --gpu_vram_gb 8 --compare both --limit 3 \
  --output_dir experiments/exp1_vram_stress

echo "=== Exp 2: λ_switch sweep (agents_fast, code_math_mix, interleave) ==="
swarm experiment --suite code_math_sweep \
  --lambda_switches 0.0,0.2,0.5,1.0 --gpu_vrams 8 \
  --agents_file configs/agents_fast.json --limit 8 \
  --output_dir experiments/exp2_switch_sweep

echo "=== Exp 3a: Interleave vs grouped - grouped ==="
swarm bench --benchmark code_math_mix --agents_file configs/agents_fast.json \
  --gpu_vram_gb 8 --lambda_switch 0.5 --compare both --limit 8 \
  --mix grouped --output_dir experiments/exp3_interleave_vs_grouped/grouped

echo "=== Exp 3b: Interleave vs grouped - interleave ==="
swarm bench --benchmark code_math_mix --agents_file configs/agents_fast.json \
  --gpu_vram_gb 8 --lambda_switch 0.5 --compare both --limit 8 \
  --mix interleave --output_dir experiments/exp3_interleave_vs_grouped/interleave

echo "=== Exp 4: Specialist routing (agents_heavy, gpu=24) ==="
swarm bench --benchmark workflowbench --agents_file configs/agents_heavy.json \
  --gpu_vram_gb 24 --compare both --limit 3 \
  --output_dir experiments/exp4_specialist_routing

echo "=== Exp 5: λ_token sweep ==="
for lt in 0.1 0.5 1.0; do
  echo "  lambda_token=$lt"
  swarm bench --benchmark workflowbench --agents_file configs/agents_fast.json \
    --gpu_vram_gb 8 --lambda_token "$lt" --compare both --limit 2 \
    --output_dir "experiments/exp5_token_sweep/lt${lt}"
done

echo "=== Full sweep (optional) ==="
echo "  python scripts/run_full_sweep.py  # timestamped runs/<ts>_full_sweep/ + meta_report.json"
echo ""
echo "=== All experiments completed ==="
