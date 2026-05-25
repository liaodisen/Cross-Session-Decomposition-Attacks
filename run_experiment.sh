#!/usr/bin/env bash
set -euo pipefail

mkdir -p intermediate_results results logs

if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  . .venv/bin/activate
fi

DOMAIN="${DOMAIN:-all}"
INPUT_FILE="${INPUT_FILE:-intent/harmful_intents.json}"
GENERATED_FILE="${GENERATED_FILE:-}"
RUN_GENERATION="${RUN_GENERATION:-auto}"
VICTIM_PANEL="${VICTIM_PANEL:-all}"
LOCAL_TP_SIZE="${LOCAL_TP_SIZE:-2}"
API_TP_SIZE="${API_TP_SIZE:-1}"
MAX_TOKENS="${MAX_TOKENS:-5192}"

domain_slug() {
  echo "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9_.-]+/_/g; s/^[._-]+//; s/[._-]+$//'
}

DOMAIN_SLUG="$(domain_slug "${DOMAIN}")"
if [ -z "${DOMAIN_SLUG}" ]; then
  DOMAIN_SLUG="all"
fi
if [ -z "${GENERATED_FILE}" ]; then
  GENERATED_FILE="intermediate_results/generated_${DOMAIN_SLUG}.json"
fi

maybe_generate() {
  case "${RUN_GENERATION}" in
    1|true|yes)
      bash run_generate.sh --input-file "${INPUT_FILE}" --domain "${DOMAIN}" --output-file "${GENERATED_FILE}" --tensor-parallel-size "${LOCAL_TP_SIZE}"
      ;;
    auto)
      if [ ! -f "${GENERATED_FILE}" ]; then
        bash run_generate.sh --input-file "${INPUT_FILE}" --domain "${DOMAIN}" --output-file "${GENERATED_FILE}" --tensor-parallel-size "${LOCAL_TP_SIZE}"
      fi
      ;;
    0|false|no)
      if [ ! -f "${GENERATED_FILE}" ]; then
        echo "Missing generated file: ${GENERATED_FILE}" >&2
        echo "Set RUN_GENERATION=1 or run run_generate.sh first." >&2
        exit 1
      fi
      ;;
    *)
      echo "RUN_GENERATION must be auto, 1, or 0" >&2
      exit 1
      ;;
  esac
}

run_victim() {
  local model="$1"
  local provider="$2"
  local tp_size="$3"

  echo "Running victim model: ${model} (${provider})"
  bash run_evaluate.sh \
    --generated-file "${GENERATED_FILE}" \
    --victim-model "${model}" \
    --provider "${provider}" \
    --tensor-parallel-size "${tp_size}" \
    --max-tokens "${MAX_TOKENS}" \
    --domain "${DOMAIN}"
}

run_qwen_panel() {
  run_victim "/model-weights/Qwen3-0.6B" vllm "${LOCAL_TP_SIZE}"
  run_victim "/model-weights/Qwen3-8B" vllm "${LOCAL_TP_SIZE}"
  run_victim "/model-weights/Qwen3-32B" vllm "${LOCAL_TP_SIZE}"
}

run_gemma_panel() {
  run_victim "/model-weights/gemma-3-1b-it" vllm "${LOCAL_TP_SIZE}"
  run_victim "/model-weights/gemma-3-12b-it" vllm "${LOCAL_TP_SIZE}"
  run_victim "/model-weights/gemma-3-27b-it" vllm "${LOCAL_TP_SIZE}"
}

run_api_anchors() {
  run_victim "deepseek-chat" deepseek "${API_TP_SIZE}"
  run_victim "gpt-5.2" openai "${API_TP_SIZE}"
}

maybe_generate

case "${VICTIM_PANEL}" in
  all)
    run_gemma_panel
    run_qwen_panel
    run_api_anchors
    ;;
  gemma)
    run_gemma_panel
    run_api_anchors
    ;;
  qwen)
    run_qwen_panel
    run_api_anchors
    ;;
  anchors)
    run_api_anchors
    ;;
  *)
    echo "VICTIM_PANEL must be one of: all, gemma, qwen, anchors" >&2
    exit 1
    ;;
esac

echo "Final answer files are under results/final_answers_*_${DOMAIN_SLUG}.json"
