#!/usr/bin/env bash
set -euo pipefail

INPUT_FILE="intent/harmful_intents.json"
DOMAIN="all"
OUTPUT_FILE=""
MODEL="huihui-ai/Qwen2.5-14B-Instruct-abliterated-v2"
TENSOR_PARALLEL_SIZE=2
BATCH_SIZE=50

usage() {
  cat <<'EOF'
Usage:
  bash run_generate.sh [options]

Options:
  --input-file <path>           Input intents JSON. Default: intent/harmful_intents.json
  --domain <name|all>           Domain filter for generation. Default: all
  --output-file <path>          Output JSON path. Default: intermediate_results/generated_<domain>.json
  --model <name-or-path>        Fixed decomposer model.
  --tensor-parallel-size <int>  Tensor parallel size for vLLM.
  --batch-size <int>            vLLM generation batch size.
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input-file)
      INPUT_FILE="$2"
      shift 2
      ;;
    --domain)
      DOMAIN="$2"
      shift 2
      ;;
    --output-file)
      OUTPUT_FILE="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --tensor-parallel-size)
      TENSOR_PARALLEL_SIZE="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
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

DOMAIN_SLUG="$(echo "$DOMAIN" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9_.-]+/_/g; s/^[._-]+//; s/[._-]+$//')"
if [[ -z "$DOMAIN_SLUG" ]]; then
  DOMAIN_SLUG="all"
fi

mkdir -p intermediate_results
if [[ -z "$OUTPUT_FILE" ]]; then
  OUTPUT_FILE="intermediate_results/generated_${DOMAIN_SLUG}.json"
fi

python blackbox/generate_questions.py \
  --input_file "$INPUT_FILE" \
  --output_file "$OUTPUT_FILE" \
  --domain "$DOMAIN" \
  --model "$MODEL" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --batch-size "$BATCH_SIZE"

echo "Generated questions saved to: $OUTPUT_FILE"
