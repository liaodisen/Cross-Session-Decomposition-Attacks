# Pairwise Decomposition Experiment

## Pipeline

The reported setup fixes the decomposition and composition stages and varies only the victim model that answers the neutralized subquestions.
The included runner scripts reproduce the Qwen3 and Gemma3 panels with `deepseek-chat` and `gpt-5.2` as API anchors.

1. `blackbox/generate_questions.py` uses `huihui-ai/Qwen2.5-14B-Instruct-abliterated-v2` to produce five neutralized subquestions per intent. The default `intent/harmful_intents.json` file contains the 600-intent evaluation set.
2. `blackbox/answer.py` sends the same subquestions to each victim model.
3. `blackbox/compose.py` recomposes each victim model's answers into one final answer per original intent.
4. `blackbox/pair_eval.py` asks a judge model to compare two final answers for the same intent.
5. `blackbox/analyze_pair_judge_agreement.py` aggregates several judge files by strict majority vote.
6. `blackbox/refusal_eval.py` judges whether each intermediate victim-model answer is a total refusal, partial refusal, or no refusal.

## Requirements

- Python 3.10+
- `vllm`, `openai`, and `tqdm`
- Local model weights for the open-weight victim and judge models used in the scripts
- `OPENAI_API_KEY` for OpenAI victims or judges
- `DEEPSEEK_API_KEY` for the `deepseek-chat` victim

Example setup:

```bash
python -m venv .venv
. .venv/bin/activate
pip install vllm openai tqdm
export OPENAI_API_KEY=...
export DEEPSEEK_API_KEY=...
```

## Run The Experiment

Generate neutralized decompositions:

```bash
bash run_generate.sh \
  --input-file intent/harmful_intents.json \
  --domain all \
  --output-file intermediate_results/generated_all.json \
  --tensor-parallel-size 2
```

Run all reported victim models and compose final answers:

```bash
RUN_GENERATION=0 \
VICTIM_PANEL=all \
DOMAIN=all \
GENERATED_FILE=intermediate_results/generated_all.json \
bash run_experiment.sh
```

Run one victim model manually:

```bash
bash run_evaluate.sh \
  --generated-file intermediate_results/generated_all.json \
  --victim-model /model-weights/Qwen3-8B \
  --provider vllm \
  --tensor-parallel-size 2 \
  --max-tokens 5192 \
  --domain all
```

Run pairwise comparisons with an OpenAI judge:

```bash
JUDGE_LABEL=gpt5.4 \
JUDGE_MODEL=gpt-5.4 \
PANEL=all \
DOMAIN=all \
bash run_pair_eval.sh
```

Run pairwise comparisons with a local vLLM judge:

```bash
JUDGE_LABEL=llama3.1-70b \
JUDGE_MODEL_PATH=/model-weights/Meta-Llama-3.1-70B-Instruct \
JUDGE_TP_SIZE=4 \
PANEL=all \
DOMAIN=all \
bash run_pair_eval_vllm_judge.sh
```

Other local judges can use the same script:

```bash
JUDGE_LABEL=qwen2.5-72b-instruct \
JUDGE_MODEL_PATH=/model-weights/Qwen2.5-72B-Instruct \
JUDGE_TP_SIZE=4 \
bash run_pair_eval_vllm_judge.sh
```

Aggregate multi-judge majority votes:

```bash
JUDGE_LABELS="gpt5.4 gpt4o llama3.1-70b qwen2.5-72b-instruct qwen3.5-122b-a10b-fp8" \
PANEL=all \
bash run_pair_judge_agreement.sh
```

Run refusal judgment on intermediate answers:

```bash
JUDGE_LABEL=llama3.1-70b \
JUDGE_MODEL=/model-weights/Meta-Llama-3.1-70B-Instruct \
TENSOR_PARALLEL_SIZE=4 \
DOMAIN=all \
bash run_refusal_eval.sh
```

Scheduler wrappers with `#SBATCH` headers are kept locally under `sbatch/`. That directory is gitignored, so these cluster-specific drivers remain available without becoming part of the tracked no-SBATCH runner surface.

## Useful Controls

- `PANEL=gemma|qwen|all` selects model-pair groups for pairwise evaluation.
- `LIMIT=<n>` limits pairwise comparisons for a smoke test.
- `DOMAIN=<name|all>` filters intents and final-answer records.
- `JUDGE_PROVIDER=openai|vllm|openai-compatible` controls the judge backend for `run_pair_eval.sh`.
- `JUDGE_BASE_URL=http://127.0.0.1:8000/v1` points `run_pair_eval.sh` at an existing OpenAI-compatible judge endpoint.
- `RUN_GENERATION=auto|1|0` controls whether `run_experiment.sh` regenerates decompositions.

## Outputs

- Generated decompositions: `intermediate_results/generated_all.json`
- Victim answers: `intermediate_results/answers_<model>_all.json`
- Composed final answers: `results/final_answers_<model>_all.json`
- Single-judge pairwise files: `results/pair_eval_<judge>_<model_a>_vs_<model_b>.json`
- Multi-judge agreement files: `results/pair_judge_agreement_<panel>_<model_a>_vs_<model_b>.json`
- Refusal judgment files: `results/refusal_eval_<judge>_<domain>.json` plus optional TSV sidecars
