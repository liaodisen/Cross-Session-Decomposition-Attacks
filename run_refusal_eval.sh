#!/usr/bin/env bash
set -euo pipefail

mkdir -p results logs

if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  . .venv/bin/activate
fi

JUDGE_LABEL="${JUDGE_LABEL:-llama3.1-70b}"
JUDGE_MODEL="${JUDGE_MODEL:-/model-weights/Meta-Llama-3.1-70B-Instruct}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
DOMAIN="${DOMAIN:-all}"
OUTPUT="${OUTPUT:-results/refusal_eval_${JUDGE_LABEL}_${DOMAIN}.json}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LIMIT="${LIMIT:-}"
RESUME="${RESUME:-1}"
SAVE_JUDGMENTS_TSV="${SAVE_JUDGMENTS_TSV:-1}"

cmd=(
  python blackbox/refusal_eval.py
  --judge-model "${JUDGE_MODEL}"
  --judge-label "${JUDGE_LABEL}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --domain "${DOMAIN}"
  --output "${OUTPUT}"
  --batch-size "${BATCH_SIZE}"
)

if [ -n "${LIMIT}" ]; then
  cmd+=(--limit "${LIMIT}")
fi
if [ "${RESUME}" = "1" ]; then
  cmd+=(--resume)
fi
if [ "${SAVE_JUDGMENTS_TSV}" = "1" ]; then
  cmd+=(--save-judgments-tsv)
fi

"${cmd[@]}"
