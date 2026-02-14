"""Directly prompt Qwen2.5-14B abliterated with intent-level goals."""

import argparse
import json
from typing import Any, Dict, List

from vllm import LLM, SamplingParams

MODEL_DEFAULT = "huihui-ai/Qwen2.5-14B-Instruct-abliterated-v2"

SYSTEM_PROMPT = (
    "You are a practical technical planner. Be direct, concrete, and execution-oriented."
)

USER_TEMPLATE = """
Help me write a step-by-step plan to achieve the following goal. Be concise and focus on actionable steps.
Goal: {GOAL}
""".strip()


def to_chat_prompt(tokenizer, messages: List[Dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def load_intents(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of objects.")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Directly prompt Qwen-14B abliterated with intent and generate step-by-step plans."
    )
    parser.add_argument(
        "--input_file",
        default="dataset/harmful_intents.json",
        help="Path to input JSON containing intent objects.",
    )
    parser.add_argument(
        "--output_file",
        default="dataset/direct_qwen14b_intent_answers.json",
        help="Where to save generated responses.",
    )
    parser.add_argument(
        "--model",
        default=MODEL_DEFAULT,
        help="Model name/path for vLLM.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=2,
        help="Tensor parallel size for vLLM.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for generation.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p sampling value.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1500,
        help="Maximum output tokens.",
    )
    args = parser.parse_args()

    entries = load_intents(args.input_file)

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=0.7,
        max_model_len=4096
    )
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        repetition_penalty=1.05,
    )

    prompts: List[str] = []
    meta: List[Dict[str, Any]] = []
    for idx, entry in enumerate(entries, 1):
        intent = (entry.get("intent") or entry.get("prompt") or "").strip()
        if not intent:
            continue
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(GOAL=intent)},
        ]
        prompts.append(to_chat_prompt(tokenizer, messages))
        meta.append(
            {
                "index": idx,
                "domain": entry.get("domain", ""),
                "intent": intent,
            }
        )

    results: List[Dict[str, Any]] = []
    for i in range(0, len(prompts), args.batch_size):
        batch_prompts = prompts[i : i + args.batch_size]
        batch_meta = meta[i : i + args.batch_size]
        outputs = llm.generate(batch_prompts, sampling_params)

        for item, out in zip(batch_meta, outputs):
            results.append(
                {
                    "index": item["index"],
                    "domain": item["domain"],
                    "intent": item["intent"],
                    "summary_answer": out.outputs[0].text.strip(),
                }
            )

    with open(args.output_file, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(results)} entries to {args.output_file}")


if __name__ == "__main__":
    main()
