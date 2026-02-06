# Blackbox Pipeline Guide

This README explains how to use the scripts under `blackbox/` to generate neutralized research questions, answer them with local/OpenAI models, compose final manuals, and score harmfulness.

## Prerequisites
- Python 3.10+ with `vllm`, `openai`, `tqdm`, `argparse` (install via `pip install -r requirements.txt` if available, or `pip install vllm openai tqdm`).
- GPU with enough memory for Qwen models when using the vLLM paths in the scripts.
- Set `OPENAI_API_KEY` in your environment before any script that calls OpenAI.
- The scripts set `HF_HOME=/scratch/disen` and download weights to `/scratch/disen` by default; override these env vars if you want a different cache/download location.

## Expected Data Layout
- Input prompts live in `dataset/testdata.json` by default. Format: list of objects with `prompt` (string) and optional `target`.
- Generated artifacts are written back to `dataset/` unless you pass explicit paths.

## End-to-End Quickstart
Run the steps from the repo root (`qwen14b_vllm/`). Each command accepts flags to override paths/models.

```bash
# 1) Generate two-round, neutralized question sets from raw prompts
python blackbox/generate_questions.py \
  --input_file dataset/testdata.json \
  --output_file dataset/generated_from_testdata.json

# 2) Answer the neutralized questions (vLLM backend by default)
python blackbox/answer.py \
  --provider vllm \
  --model Qwen2.5-7B-Instruct \
  --input dataset/generated_from_testdata.json \
  --output dataset/answers_qwen2.5-7b.json
#    To use OpenAI instead: --provider openai --model gpt-4.1-mini --max-tokens 1500

# 3) Compose final action-oriented manuals from the Q&A batches
python blackbox/compose.py \
  --inputs dataset/answers_qwen2.5-7b.json \
  --outputs result/summary_qwen2.5-7b.json \
  --tensor-parallel-size 1

# 4) Score harmfulness of any dense paragraphs (e.g., composed summaries)
python blackbox/evaluate_harmness.py \
  --input result/summary_qwen2.5-7b.json \
  --output dataset/harm_eval_results.json \
  --model gpt-4.1-mini

# 5) Aggregate and inspect the harm scores
python blackbox/analyze_eval.py \
  --input dataset/harm_eval_results.json \
  --output dataset/harm_eval_summary.json
```

## Script Reference
- `blackbox/generate_questions.py`
  - Two-pass question generation with Qwen2.5-14B via vLLM.
  - Flags: `--input_file`, `--output_file`. Batch size is fixed at 50; sampling params for exploration + neutralization are defined in the script.
- `blackbox/answer.py`
  - Turns neutralized questions into answers. Choose backend with `--provider {vllm,openai}`.
  - vLLM: uses `--model` (or `--model-path`) and `--tensor-parallel-size`. Weights default to `/model-weights/<model>`.
  - OpenAI: honors `--model` and `--max-tokens`; requires `OPENAI_API_KEY`.
  - Output default: `dataset/answers_<model>.json`.
- `blackbox/compose.py`
  - Converts batches of Q&A pairs into an action-focused manual per intent.
  - Flags: `--inputs` and `--outputs` (lists of equal length), `--tensor-parallel-size`.
  - Uses Qwen2.5-14B via vLLM with a composing prompt defined in `blackbox/prompt.py`.
- `blackbox/evaluate_harmness.py`
  - Scores paragraphs on six axes plus composite harm level using an OpenAI model.
  - Flags: `--input`, `--output`, `--model`, `--prompt_file` (to override the default rubric prompt).
- `blackbox/analyze_eval.py`
  - Summarizes harm evaluation JSON: mean per-axis, raw score stats, harm level distribution, and top/bottom examples.
  - Flags: `--input`, optional `--output` (otherwise prints to stdout).

## Tips
- Files contain hard-coded API keys in the repo; prefer setting `OPENAI_API_KEY` in your shell to avoid using the placeholder value.
- Adjust `tensor_parallel_size`, `gpu_memory_utilization`, and `max_model_len` in scripts if your GPU memory differs.
- If you change prompt wording, edit `blackbox/prompt.py` and rerun the pipeline from the generation step that depends on it.
