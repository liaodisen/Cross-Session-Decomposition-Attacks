#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs results

if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  . .venv/bin/activate
fi

JUDGE_PROVIDER="${JUDGE_PROVIDER:-openai}"
JUDGE_LABEL="${JUDGE_LABEL:-gpt5.4}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5.4}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-}"
JUDGE_API_KEY="${JUDGE_API_KEY:-}"
JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-160}"
JUDGE_TEMPERATURE="${JUDGE_TEMPERATURE:-0.0}"
DOMAIN="${DOMAIN:-all}"
PANEL="${PANEL:-all}"
LIMIT="${LIMIT:-}"

GEMMA_MODEL_LABELS=(
  "gemma3-1b"
  "gemma3-12b"
  "gemma3-27b"
  "deepseek-chat"
  "chatgpt5.2"
)

GEMMA_MODEL_FILES=(
  "results/final_answers_model-weights_gemma-3-1b-it_all.json"
  "results/final_answers_model-weights_gemma-3-12b-it_all.json"
  "results/final_answers_model-weights_gemma-3-27b-it_all.json"
  "results/final_answers_deepseek-chat_all.json"
  "results/final_answers_gpt-5.2_all.json"
)

QWEN_MODEL_LABELS=(
  "qwen3-0.6b"
  "qwen3-8b"
  "qwen3-32b"
  "deepseek-chat"
  "chatgpt5.2"
)

QWEN_MODEL_FILES=(
  "results/final_answers_model-weights_Qwen3-0.6B_all.json"
  "results/final_answers_model-weights_Qwen3-8B_all.json"
  "results/final_answers_model-weights_Qwen3-32B_all.json"
  "results/final_answers_deepseek-chat_all.json"
  "results/final_answers_gpt-5.2_all.json"
)

slugify() {
  echo "$1" | sed -E 's/[^A-Za-z0-9_.-]+/_/g; s/^[._-]+//; s/[._-]+$//'
}

validate_files() {
  local files=("$@")
  local model_file
  for model_file in "${files[@]}"; do
    if [ ! -f "${model_file}" ]; then
      echo "Missing required input file: ${model_file}" >&2
      echo "Run run_evaluate.sh or run_experiment.sh to create final answer files first." >&2
      exit 1
    fi
  done
}

run_pair() {
  local panel_name="$1"
  local label_a="$2"
  local input_a="$3"
  local label_b="$4"
  local input_b="$5"
  local judge_slug output
  judge_slug="$(slugify "${JUDGE_LABEL}")"
  output="results/pair_eval_${judge_slug}_${label_a}_vs_${label_b}.json"

  echo "Running ${panel_name} pair eval with judge=${JUDGE_LABEL}: ${label_a} vs ${label_b}"

  local cmd=(
    python blackbox/pair_eval.py
    --input-a "${input_a}"
    --input-b "${input_b}"
    --label-a "${label_a}"
    --label-b "${label_b}"
    --output "${output}"
    --judge-provider "${JUDGE_PROVIDER}"
    --judge-model "${JUDGE_MODEL}"
    --judge-max-tokens "${JUDGE_MAX_TOKENS}"
    --judge-temperature "${JUDGE_TEMPERATURE}"
    --domain "${DOMAIN}"
    --save-comparisons
  )

  if [ -n "${JUDGE_BASE_URL}" ]; then
    cmd+=(--judge-base-url "${JUDGE_BASE_URL}")
  fi
  if [ -n "${JUDGE_API_KEY}" ]; then
    cmd+=(--judge-api-key "${JUDGE_API_KEY}")
  fi
  if [ -n "${LIMIT}" ]; then
    cmd+=(--limit "${LIMIT}")
  fi

  "${cmd[@]}"
}

run_panel() {
  local panel_name="$1"
  local labels=()
  local files=()

  case "${panel_name}" in
    gemma)
      labels=("${GEMMA_MODEL_LABELS[@]}")
      files=("${GEMMA_MODEL_FILES[@]}")
      ;;
    qwen)
      labels=("${QWEN_MODEL_LABELS[@]}")
      files=("${QWEN_MODEL_FILES[@]}")
      ;;
    *)
      echo "Unknown panel: ${panel_name}" >&2
      exit 1
      ;;
  esac

  validate_files "${files[@]}"

  local model_count=${#labels[@]}
  for ((i = 0; i < model_count; i++)); do
    for ((j = i + 1; j < model_count; j++)); do
      run_pair "${panel_name}" "${labels[i]}" "${files[i]}" "${labels[j]}" "${files[j]}"
    done
  done
}

case "${PANEL}" in
  all)
    run_panel gemma
    run_panel qwen
    ;;
  gemma|qwen)
    run_panel "${PANEL}"
    ;;
  *)
    echo "PANEL must be one of: all, gemma, qwen" >&2
    exit 1
    ;;
esac
