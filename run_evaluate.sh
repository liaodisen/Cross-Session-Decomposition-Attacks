#!/usr/bin/env bash
set -euo pipefail

VICTIM_MODEL="huihui-ai/Qwen2.5-14B-Instruct-abliterated-v2"
PROVIDER="vllm"
DOMAIN="all"
GENERATED_FILE=""
TENSOR_PARALLEL_SIZE=2
MAX_TOKENS=2048

usage() {
  cat <<'EOF'
Usage:
  bash run_evaluate.sh [options]

Options:
  --victim-model <name>         Victim model for answer stage.
  --provider <vllm|openai|deepseek>
  --domain <name|all>           Domain filter applied to answer/compose.
  --generated-file <path>       Existing generated questions JSON.
                                Default: intermediate_results/generated_<domain>.json
  --tensor-parallel-size <int>  Tensor parallel size for vLLM stages.
  --max-tokens <int>            Max tokens for answer stage.
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --victim-model)
      VICTIM_MODEL="$2"
      shift 2
      ;;
    --provider)
      PROVIDER="$2"
      shift 2
      ;;
    --domain)
      DOMAIN="$2"
      shift 2
      ;;
    --generated-file)
      GENERATED_FILE="$2"
      shift 2
      ;;
    --tensor-parallel-size)
      TENSOR_PARALLEL_SIZE="$2"
      shift 2
      ;;
    --max-tokens)
      MAX_TOKENS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

mkdir -p intermediate_results results

MODEL_SLUG="$(echo "$VICTIM_MODEL" | sed -E 's/[^A-Za-z0-9_.-]+/_/g; s/^[._-]+//; s/[._-]+$//')"
DOMAIN_SLUG="$(echo "$DOMAIN" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9_.-]+/_/g; s/^[._-]+//; s/[._-]+$//')"
if [[ -z "$DOMAIN_SLUG" ]]; then
  DOMAIN_SLUG="all"
fi

ANSWER_FILE="intermediate_results/answers_${MODEL_SLUG}_${DOMAIN_SLUG}.json"
FINAL_ANSWER_FILE="results/final_answers_${MODEL_SLUG}_${DOMAIN_SLUG}.json"
if [[ -z "$GENERATED_FILE" ]]; then
  GENERATED_FILE="intermediate_results/generated_${DOMAIN_SLUG}.json"
fi

echo "[1/2] Answering neutralized questions with victim model..."
python blackbox/answer.py \
  --provider "$PROVIDER" \
  --model "$VICTIM_MODEL" \
  --max-tokens "$MAX_TOKENS" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --input "$GENERATED_FILE" \
  --output "$ANSWER_FILE" \
  --domain "$DOMAIN"

echo "[2/2] Composing final answers..."
python blackbox/compose.py \
  --inputs "$ANSWER_FILE" \
  --outputs "$FINAL_ANSWER_FILE" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --domain "$DOMAIN"

echo "Done."
echo "Intermediate:"
echo "  $GENERATED_FILE"
echo "  $ANSWER_FILE"
echo "Results:"
echo "  $FINAL_ANSWER_FILE"
