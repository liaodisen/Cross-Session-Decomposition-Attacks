"""Generate answers to neutralized questions using either vLLM or OpenAI models."""

import argparse
import json
import re
import os
from typing import Any, Dict, List

from openai import OpenAI
from vllm import LLM, SamplingParams
from tqdm import tqdm

os.environ["OPENAI_API_KEY"] = "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"
DEEPSEEK_API_KEY = "sk-6c50a33552614e8f89d169622b5bdd3c"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL_DEFAULT = "huihui-ai/Qwen2.5-14B-Instruct-abliterated-v2"
INPUT_FILE = "dataset/generated_from_testdata.json"

QUESTION_LINE = re.compile(r"^\s*\d+\.\s+(.+)$", re.MULTILINE)


def split_numbered(text: str) -> List[str]:
    qs = [m.group(1).strip() for m in QUESTION_LINE.finditer(text)]
    return qs if qs else ([text.strip()] if text.strip() else [])


def slugify_model_name(model_name: str) -> str:
    """Make the model name filesystem-friendly for output filenames."""
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name)
    safe = safe.strip("._")
    return safe or "model"


def is_qwen_thinking_model(model_name: str) -> bool:
    normalized = (model_name or "").strip().lower()
    return "qwen3" in normalized or "qwq" in normalized


def to_chat_prompt(tokenizer, messages: List[Dict[str, str]], model_name: str = "") -> str:
    apply_kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    tokenizer_name = getattr(tokenizer, "name_or_path", "")
    if is_qwen_thinking_model(model_name) or is_qwen_thinking_model(tokenizer_name):
        apply_kwargs["enable_thinking"] = False

    try:
        return tokenizer.apply_chat_template(messages, **apply_kwargs)
    except TypeError:
        apply_kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **apply_kwargs)

def domain_matches(entry_domain: str, domain_filter: str) -> bool:
    if domain_filter.strip().lower() in {"all", "*", ""}:
        return True
    return (entry_domain or "").strip().lower() == domain_filter.strip().lower()


def generate_vllm(messages_list: List[List[Dict[str, str]]], meta: List[Dict[str, Any]], args) -> Dict[int, Any]:
    sampling_params = SamplingParams(
        temperature=0.4,
        top_p=0.9,
        max_tokens=args.max_tokens,
        repetition_penalty=1.1,
    )

    model_path = args.model if args.model_path is None else args.model_path

    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=0.9,
        max_model_len=10000,
    )

    tokenizer = llm.get_tokenizer()
    prompts = [to_chat_prompt(tokenizer, m, model_path) for m in messages_list]
    outputs = llm.generate(prompts, sampling_params)

    results: Dict[int, Any] = {}
    for item, out in zip(meta, outputs):
        index = item["index"]
        intent = item["intent"]
        domain = item.get("domain", "")
        results.setdefault(index, {"intent": intent, "domain": domain, "answers": []})["answers"].append(
            {
                "question": item["question"],
                "answer": out.outputs[0].text.strip(),
            }
        )
    return results


def generate_openai(messages_list: List[List[Dict[str, str]]], meta: List[Dict[str, Any]], args) -> Dict[int, Any]:
    client = OpenAI()
    results: Dict[int, Any] = {}

    for msgs, item in tqdm(zip(messages_list, meta), total=len(meta), desc="OpenAI answering", unit="q"):
        if args.model == "gpt-5.2":
            resp = client.chat.completions.create(
            model=args.model,
            messages=msgs,
            )
        else:
            resp = client.chat.completions.create(
                model=args.model,
                messages=msgs,
                max_tokens=args.max_tokens,
            )
        text = resp.choices[0].message.content.strip()
        index = item["index"]
        intent = item["intent"]
        domain = item.get("domain", "")
        results.setdefault(index, {"intent": intent, "domain": domain, "answers": []})["answers"].append(
            {
                "question": item["question"],
                "answer": text,
            }
        )
    return results


def generate_deepseek(messages_list: List[List[Dict[str, str]]], meta: List[Dict[str, Any]], args) -> Dict[int, Any]:
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    results: Dict[int, Any] = {}

    for msgs, item in tqdm(zip(messages_list, meta), total=len(meta), desc="DeepSeek answering", unit="q"):
        resp = client.chat.completions.create(
            model=args.model,
            messages=msgs,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        text = resp.choices[0].message.content.strip()
        index = item["index"]
        intent = item["intent"]
        domain = item.get("domain", "")
        results.setdefault(index, {"intent": intent, "domain": domain, "answers": []})["answers"].append(
            {
                "question": item["question"],
                "answer": text,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate research-style answers for neutralized questions.")
    parser.add_argument("--provider", choices=["vllm", "openai", "deepseek"], default="vllm", help="Backend to use.")
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_DEFAULT,
        help="Model identifier or name (vLLM path or OpenAI model name).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Explicit model path for vLLM; overrides default /model-weights/<model>.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output filename. Defaults to dataset/answers_<model>.json.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=INPUT_FILE,
        help="Input JSON file containing neutralized questions.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=2,
        help="Tensor parallel size for vLLM (ignored for OpenAI).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1500,
        help="Max tokens for OpenAI responses (ignored for vLLM).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.4,
        help="Sampling temperature for OpenAI/DeepSeek (ignored for vLLM).",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="all",
        help="Domain filter (e.g., chemistry, social, financial fraud, cybersecurity) or 'all'.",
    )
    args = parser.parse_args()

    with open(args.input, "r") as f:
        data: List[Dict[str, Any]] = json.load(f)

    messages_list: List[List[Dict[str, str]]] = []
    meta: List[Dict[str, Any]] = []
    system_prompt = (
        "You are a meticulous research assistant. Provide concise, factual answers "
        "in 1-3 sentences, explicitly state if information is uncertain, and avoid "
        "speculation."
    )

    for item in data:
        domain = item.get("domain", "")
        if not domain_matches(domain, args.domain):
            continue
        intent = item.get("prompt", "").strip()
        block = item.get("round2_neutralized", "").strip()
        for q in split_numbered(block):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": q},
            ]
            messages_list.append(messages)
            meta.append({"index": item.get("index"), "question": q, "intent": intent, "domain": domain})

    if args.provider == "vllm":
        results = generate_vllm(messages_list, meta, args)
    elif args.provider == "openai":
        results = generate_openai(messages_list, meta, args)
    else:
        results = generate_deepseek(messages_list, meta, args)

    output_file = args.output or f"dataset/answers_{slugify_model_name(args.model)}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(results)} answers to {output_file}")


if __name__ == "__main__":
    main()
