#!/usr/bin/env bash
set -euo pipefail

mkdir -p results

if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  . .venv/bin/activate
fi

PANEL="${PANEL:-all}"
RESULT_DIR="${RESULT_DIR:-results}"
JUDGE_LABELS="${JUDGE_LABELS:-gpt5.4 gpt4o llama3.1-70b qwen2.5-72b-instruct qwen3.5-122b-a10b-fp8}"
WAIT="${WAIT:-0}"
NO_VECTORS="${NO_VECTORS:-0}"

IFS=' ' read -r -a JUDGES <<< "${JUDGE_LABELS}"

GEMMA_MODEL_LABELS=(
  "gemma3-1b"
  "gemma3-12b"
  "gemma3-27b"
  "deepseek-chat"
  "chatgpt5.2"
)

QWEN_MODEL_LABELS=(
  "qwen3-0.6b"
  "qwen3-8b"
  "qwen3-32b"
  "deepseek-chat"
  "chatgpt5.2"
)

run_pair_agreement() {
  local panel_name="$1"
  local label_a="$2"
  local label_b="$3"
  local output="results/pair_judge_agreement_${panel_name}_${label_a}_vs_${label_b}.json"

  echo "Aggregating judges for ${panel_name}: ${label_a} vs ${label_b}"
  local cmd=(
    python blackbox/analyze_pair_judge_agreement.py
    --result-dir "${RESULT_DIR}"
    --judge-labels "${JUDGES[@]}"
    --model-a "${label_a}"
    --model-b "${label_b}"
    --output "${output}"
  )

  if [ "${WAIT}" = "1" ]; then
    cmd+=(--wait)
  fi
  if [ "${NO_VECTORS}" = "1" ]; then
    cmd+=(--no-vectors)
  fi

  "${cmd[@]}"
}

run_panel() {
  local panel_name="$1"
  local labels=()

  case "${panel_name}" in
    gemma)
      labels=("${GEMMA_MODEL_LABELS[@]}")
      ;;
    qwen)
      labels=("${QWEN_MODEL_LABELS[@]}")
      ;;
    *)
      echo "Unknown panel: ${panel_name}" >&2
      exit 1
      ;;
  esac

  local model_count=${#labels[@]}
  for ((i = 0; i < model_count; i++)); do
    for ((j = i + 1; j < model_count; j++)); do
      run_pair_agreement "${panel_name}" "${labels[i]}" "${labels[j]}"
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
