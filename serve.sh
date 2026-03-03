#!/bin/bash
#SBATCH --job-name=model_serve
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=vllm_%j.log

source ./.venv/bin/activate

MODEL="huihui-ai/Qwen2.5-14B-Instruct-abliterated-v2"
PORT=8000

echo "Running on node: $(hostname)"
echo "starting server for model: $MODEL on port: $PORT"

vllm serve "$MODEL" \
    --host 127.0.0.1 \
    --port $PORT \
    --tensor-parallel-size 1 \
    --max-model-len 4000 \
    --max-num-seqs 16 \
    --max-num-batched-tokens 32768 \


