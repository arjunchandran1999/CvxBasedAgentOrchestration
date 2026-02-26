#!/usr/bin/env bash
# Run experiments: LP vs LLM Planner Routing
# Execute from project root: ./scripts/run_experiments.sh
#
# Full sweep (all benchmarks, timestamped): python scripts/run_full_sweep.py
# Creates runs/<timestamp>_full_sweep/ with meta_report.json (Pareto metrics).

set -e
cd "$(dirname "$0")/.."
mkdir -p experiments runs

echo "=== Phase 0: Baseline (as-is, static estimators) ==="

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

echo "=== Exp 6: Literature-style routing suite (multi-benchmark, multi-seed) ==="
# Similar to routing papers (e.g., RouteLLM / FrugalGPT / Routoo): evaluate cost-quality tradeoffs on
# a diverse benchmark mix and compare routers to single-model baselines.
#
# Benchmarks available in this repo:
#   boolq, squad, xsum, arc, gsm8k, humaneval, mbpp, workflowbench, code_math_mix, agentic_bench
#
# We use a 2-model pool (strong+weak) to match common router paper setups.
PAIR_AGENTS="configs/agents_pair_phi4_gemma2.json"
STRONG_ONLY="configs/agents_single_phi4.json"
WEAK_ONLY="configs/agents_single_gemma2.json"

for seed in 0 1 2; do
  echo "  seed=$seed"

  # Router comparison (LP vs LLM) on a mixed suite
  swarm bench \
    --benchmark boolq --benchmark squad --benchmark xsum --benchmark arc_challenge --benchmark gsm8k \
    --agents_file "$PAIR_AGENTS" \
    --gpu_vram_gb 48 --compare both --limit 20 --seed "$seed" \
    --mix interleave --output_dir "experiments/exp6_paper_suite/static/seed${seed}" \
    --estimator_tag static

  # Code tasks (optional: enable code execution for stricter scoring)
  swarm bench \
    --benchmark humaneval --benchmark mbpp \
    --agents_file "$PAIR_AGENTS" \
    --gpu_vram_gb 48 --compare both --limit 10 --seed "$seed" \
    --mix interleave --code_eval --output_dir "experiments/exp6_paper_suite/static/seed${seed}_code" \
    --estimator_tag static

  # Single-model baselines (always-strong vs always-weak)
  swarm bench \
    --benchmark boolq --benchmark squad --benchmark xsum --benchmark arc_challenge --benchmark gsm8k \
    --agents_file "$STRONG_ONLY" \
    --gpu_vram_gb 48 --compare lp --limit 20 --seed "$seed" \
    --mix interleave --output_dir "experiments/exp6_paper_suite/baselines_strong/seed${seed}" \
    --estimator_tag static

  swarm bench \
    --benchmark boolq --benchmark squad --benchmark xsum --benchmark arc_challenge --benchmark gsm8k \
    --agents_file "$WEAK_ONLY" \
    --gpu_vram_gb 48 --compare lp --limit 20 --seed "$seed" \
    --mix interleave --output_dir "experiments/exp6_paper_suite/baselines_weak/seed${seed}" \
    --estimator_tag static
done

echo "=== Phase 1: Train estimators (quality + token multiplier) ==="
# Training requires a meaningful signal. We enable --judge so the quality estimator gets 0..1 scores,
# and token multiplier learns from actual token usage in telemetry.
TRAIN_DIR="experiments/exp0_estimator_training"
ESTIMATOR_STATE="${TRAIN_DIR}/estimator_state.json"
mkdir -p "$TRAIN_DIR"

swarm bench --benchmark workflowbench --agents_file configs/agents_heavy.json \
  --gpu_vram_gb 48 --compare lp --limit 6 --horizon_depth 3 \
  --judge --output_dir "$TRAIN_DIR" \
  --estimator_out "$ESTIMATOR_STATE" --estimator_tag trained

echo "Wrote trained estimator state to: ${ESTIMATOR_STATE}"

echo "=== Phase 2: Rerun key experiments with trained estimators (frozen) ==="

echo "=== Exp T1: VRAM stress (trained, agents_heavy, gpu=8) ==="
swarm bench --benchmark workflowbench --agents_file configs/agents_heavy.json \
  --gpu_vram_gb 8 --compare both --limit 3 \
  --output_dir experiments/expT1_vram_stress_trained \
  --estimator_in "$ESTIMATOR_STATE" --freeze_estimator --estimator_tag trained

echo "=== Exp T2: λ_switch sweep (trained, agents_fast, code_math_mix, interleave) ==="
swarm experiment --suite code_math_sweep \
  --lambda_switches 0.0,0.2,0.5,1.0 --gpu_vrams 8 \
  --agents_file configs/agents_fast.json --limit 8 \
  --output_dir experiments/expT2_switch_sweep_trained \
  --estimator_in "$ESTIMATOR_STATE" --freeze_estimator --estimator_tag trained

echo "=== Exp T3: Specialist routing (trained, agents_heavy, gpu=24) ==="
swarm bench --benchmark workflowbench --agents_file configs/agents_heavy.json \
  --gpu_vram_gb 24 --compare both --limit 3 \
  --output_dir experiments/expT3_specialist_routing_trained \
  --estimator_in "$ESTIMATOR_STATE" --freeze_estimator --estimator_tag trained

echo "=== Exp T4: Literature-style routing suite (trained+frozen, multi-benchmark, multi-seed) ==="
for seed in 0 1 2; do
  echo "  seed=$seed"
  swarm bench \
    --benchmark boolq --benchmark squad --benchmark xsum --benchmark arc_challenge --benchmark gsm8k \
    --agents_file "$PAIR_AGENTS" \
    --gpu_vram_gb 48 --compare both --limit 20 --seed "$seed" \
    --mix interleave --output_dir "experiments/expT4_paper_suite_trained/seed${seed}" \
    --estimator_in "$ESTIMATOR_STATE" --freeze_estimator --estimator_tag trained
done

echo "=== Exp 7: ALL benchmarks battery (static) ==="
# Runs all registered benchmarks to avoid cherry-picking. Some are DAG (workflowbench/agentic_bench),
# some are single-shot (boolq/squad/xsum/arc_challenge/gsm8k/humaneval/mbpp), and code benchmarks can use --code_eval.
ALL_BENCHMARKS=(boolq squad xsum arc_challenge gsm8k humaneval mbpp workflowbench code_math_mix agentic_bench)
for seed in 0 1 2; do
  echo "  seed=$seed"
  for b in "${ALL_BENCHMARKS[@]}"; do
    echo "    benchmark=$b"
    # DAG benchmarks: sweep horizon_depth to quantify lookahead effect.
    if [[ "$b" == "workflowbench" || "$b" == "agentic_bench" ]]; then
      for hd in 0 1 3; do
        swarm bench --benchmark "$b" --agents_file configs/agents_heavy.json \
          --gpu_vram_gb 48 --compare both --limit 10 --seed "$seed" \
          --horizon_depth "$hd" --output_dir "experiments/exp7_all_benchmarks/static/${b}/seed${seed}_hd${hd}" \
          --estimator_tag static
      done
    else
      EXTRA=()
      if [[ "$b" == "humaneval" || "$b" == "mbpp" ]]; then
        EXTRA=(--code_eval)
      fi
      swarm bench --benchmark "$b" --agents_file "$PAIR_AGENTS" \
        --gpu_vram_gb 48 --compare both --limit 20 --seed "$seed" \
        --mix interleave --output_dir "experiments/exp7_all_benchmarks/static/${b}/seed${seed}" \
        "${EXTRA[@]}" --estimator_tag static
    fi
  done
done

echo "=== Exp 8: Ablations (static) ==="
# Objective ablations commonly reported in routing papers/system evaluations.
# We implement them via parameter settings:
#  - no_switch: lambda_switch=0
#  - no_token: lambda_token=0
#  - quality_only: lambda_token=0, lambda_switch=0
#  - no_vram_binding: gpu_vram_gb set very large (constraint never binds)
ABLATIONS=(base no_switch no_token quality_only no_vram_binding)
for seed in 0 1 2; do
  echo "  seed=$seed"
  for ab in "${ABLATIONS[@]}"; do
    echo "    ablation=$ab"
    LT=0.5
    LS=0.2
    VR=48
    if [[ "$ab" == "no_switch" ]]; then LS=0.0; fi
    if [[ "$ab" == "no_token" ]]; then LT=0.0; fi
    if [[ "$ab" == "quality_only" ]]; then LT=0.0; LS=0.0; fi
    if [[ "$ab" == "no_vram_binding" ]]; then VR=1000000; fi

    # Evaluate on representative tasks: one QA (gsm8k), one extraction (squad), one summarize (xsum),
    # one code (humaneval), and one DAG (workflowbench).
    swarm bench --benchmark gsm8k --benchmark squad --benchmark xsum \
      --agents_file "$PAIR_AGENTS" --gpu_vram_gb "$VR" --compare both --limit 30 --seed "$seed" \
      --lambda_token "$LT" --lambda_switch "$LS" --mix interleave \
      --output_dir "experiments/exp8_ablations/static/seed${seed}/${ab}/nlp" \
      --estimator_tag static

    swarm bench --benchmark humaneval \
      --agents_file "$PAIR_AGENTS" --gpu_vram_gb "$VR" --compare both --limit 15 --seed "$seed" \
      --lambda_token "$LT" --lambda_switch "$LS" --mix interleave --code_eval \
      --output_dir "experiments/exp8_ablations/static/seed${seed}/${ab}/code" \
      --estimator_tag static

    for hd in 0 1 3; do
      swarm bench --benchmark workflowbench --agents_file configs/agents_heavy.json \
        --gpu_vram_gb "$VR" --compare both --limit 10 --seed "$seed" \
        --lambda_token "$LT" --lambda_switch "$LS" --horizon_depth "$hd" \
        --output_dir "experiments/exp8_ablations/static/seed${seed}/${ab}/workflowbench_hd${hd}" \
        --estimator_tag static
    done
  done
done

echo "=== Exp T5: ALL benchmarks battery (trained+frozen) ==="
for seed in 0 1 2; do
  echo "  seed=$seed"
  for b in "${ALL_BENCHMARKS[@]}"; do
    echo "    benchmark=$b"
    if [[ "$b" == "workflowbench" || "$b" == "agentic_bench" ]]; then
      for hd in 0 1 3; do
        swarm bench --benchmark "$b" --agents_file configs/agents_heavy.json \
          --gpu_vram_gb 48 --compare both --limit 10 --seed "$seed" \
          --horizon_depth "$hd" --output_dir "experiments/expT5_all_benchmarks_trained/${b}/seed${seed}_hd${hd}" \
          --estimator_in "$ESTIMATOR_STATE" --freeze_estimator --estimator_tag trained
      done
    else
      EXTRA=()
      if [[ "$b" == "humaneval" || "$b" == "mbpp" ]]; then
        EXTRA=(--code_eval)
      fi
      swarm bench --benchmark "$b" --agents_file "$PAIR_AGENTS" \
        --gpu_vram_gb 48 --compare both --limit 20 --seed "$seed" \
        --mix interleave --output_dir "experiments/expT5_all_benchmarks_trained/${b}/seed${seed}" \
        "${EXTRA[@]}" --estimator_in "$ESTIMATOR_STATE" --freeze_estimator --estimator_tag trained
    fi
  done
done

echo "=== Full sweep (optional) ==="
echo "  python scripts/run_full_sweep.py  # timestamped runs/<ts>_full_sweep/ + meta_report.json"
echo "  python scripts/run_full_sweep.py --estimator_in \"$ESTIMATOR_STATE\" --freeze_estimator --estimator_tag trained"
echo ""
echo "=== All experiments completed ==="
