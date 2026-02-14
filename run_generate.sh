#!/usr/bin/env bash
set -euo pipefail

INPUT_FILE="dataset/harmful_intents.json"
DOMAIN="all"
OUTPUT_FILE=""

usage() {
  cat <<'EOF'
Usage:
  bash run_generate.sh [options]

Options:
  --input-file <path>   Input intents JSON. Default: dataset/harmful_intents.json
  --domain <name|all>   Domain filter for generation. Default: all
  --output-file <path>  Output JSON path. Default: intermediate_results/generated_<domain>.json
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
  --domain "$DOMAIN"

echo "Generated questions saved to: $OUTPUT_FILE"
