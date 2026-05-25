#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs results

if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  . .venv/bin/activate
fi

JUDGE_LABEL="${JUDGE_LABEL:-llama3.1-70b}"
JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-/model-weights/Meta-Llama-3.1-70B-Instruct}"
JUDGE_PORT="${JUDGE_PORT:-8000}"
JUDGE_TP_SIZE="${JUDGE_TP_SIZE:-4}"
JUDGE_MAX_MODEL_LEN="${JUDGE_MAX_MODEL_LEN:-10000}"
JUDGE_GPU_MEMORY_UTILIZATION="${JUDGE_GPU_MEMORY_UTILIZATION:-0.9}"
JUDGE_MAX_NUM_SEQS="${JUDGE_MAX_NUM_SEQS:-16}"
JUDGE_MAX_NUM_BATCHED_TOKENS="${JUDGE_MAX_NUM_BATCHED_TOKENS:-32768}"
VLLM_STARTUP_TIMEOUT="${VLLM_STARTUP_TIMEOUT:-3600}"
VLLM_BASE_URL="http://127.0.0.1:${JUDGE_PORT}/v1"
VLLM_PID=""

slugify() {
  echo "$1" | sed -E 's/[^A-Za-z0-9_.-]+/_/g; s/^[._-]+//; s/[._-]+$//'
}

cleanup_vllm() {
  if [[ -n "${VLLM_PID:-}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
  fi
  VLLM_PID=""
}
trap cleanup_vllm EXIT

wait_for_vllm() {
  local base_url="$1"
  local pid="$2"
  local deadline=$((SECONDS + VLLM_STARTUP_TIMEOUT))

  until python - "$base_url" <<'PY'
import json
import sys
import urllib.request

base_url = sys.argv[1].rstrip("/")
try:
    with urllib.request.urlopen(f"{base_url}/models", timeout=5) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    raise SystemExit(0 if data.get("data") else 1)
except Exception:
    raise SystemExit(1)
PY
  do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "vLLM judge server exited before it was ready." >&2
      return 1
    fi
    if (( SECONDS >= deadline )); then
      echo "Timed out waiting for vLLM judge server at ${base_url}." >&2
      return 1
    fi
    sleep 10
  done
}

start_vllm_judge() {
  local judge_slug log_file
  judge_slug="$(slugify "${JUDGE_LABEL}")"
  log_file="logs/vllm_judge_${judge_slug}_manual.log"

  echo "Starting vLLM judge ${JUDGE_LABEL}: ${JUDGE_MODEL_PATH}"
  vllm serve "${JUDGE_MODEL_PATH}" \
    --served-model-name "${JUDGE_LABEL}" \
    --host 127.0.0.1 \
    --port "${JUDGE_PORT}" \
    --tensor-parallel-size "${JUDGE_TP_SIZE}" \
    --max-model-len "${JUDGE_MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${JUDGE_GPU_MEMORY_UTILIZATION}" \
    --max-num-seqs "${JUDGE_MAX_NUM_SEQS}" \
    --max-num-batched-tokens "${JUDGE_MAX_NUM_BATCHED_TOKENS}" \
    > "${log_file}" 2>&1 &

  VLLM_PID=$!
  if ! wait_for_vllm "${VLLM_BASE_URL}" "${VLLM_PID}"; then
    tail -n 80 "${log_file}" >&2 || true
    exit 1
  fi
}

start_vllm_judge

JUDGE_PROVIDER=vllm \
JUDGE_LABEL="${JUDGE_LABEL}" \
JUDGE_MODEL="${JUDGE_LABEL}" \
JUDGE_BASE_URL="${VLLM_BASE_URL}" \
bash run_pair_eval.sh
