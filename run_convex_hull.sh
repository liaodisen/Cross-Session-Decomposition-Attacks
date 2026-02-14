#!/usr/bin/env bash
set -uo pipefail

# Optional overrides:
#   N=3 SEED=42 ./run_convex_hull.sh
N="${N:-3}"
SEED="${SEED:-42}"

SCRIPT_PATH="convex_approximate/convex_hull_analysis.py"
OUT_DIR="convex_approximate_result"
mkdir -p "$OUT_DIR"

MODELS=(
  "Qwen/Qwen2.5-0.5B-Instruct"
  "Qwen/Qwen2.5-3B-Instruct"
)

NUM_SAMPLES_LIST=(3 5 10 15 20)

for model in "${MODELS[@]}"; do
  model_slug="${model//\//_}"
  for num_samples in "${NUM_SAMPLES_LIST[@]}"; do
    run_name="${model_slug}_samples${num_samples}"
    results_json="${OUT_DIR}/${run_name}.json"
    log_file="${OUT_DIR}/${run_name}.log"

    echo "============================================================"
    echo "Running model=${model} num_samples=${num_samples}"
    echo "Results: ${results_json}"
    echo "Log:     ${log_file}"
    echo "============================================================"

    python "$SCRIPT_PATH" \
      --n "$N" \
      --seed "$SEED" \
      --llm-model-name "$model" \
      --num-samples "$num_samples" \
      --results-json "$results_json" \
      2>&1 | tee "$log_file"
  done
done
