#!/usr/bin/env python3
"""Run inference against a locally served vLLM OpenAI-compatible endpoint."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import requests


def discover_model(base_url: str, timeout: float) -> str:
    resp = requests.get(f"{base_url}/models", timeout=timeout)
    resp.raise_for_status()
    data = resp.json().get("data", [])
    if not data:
        raise RuntimeError("No models found at /models endpoint.")
    return data[0]["id"]


def infer_one(
    base_url: str,
    model: str,
    prompt: str,
    system_prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout: float,
) -> Dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    resp = requests.post(f"{base_url}/chat/completions", json=payload, timeout=timeout)
    resp.raise_for_status()
    body = resp.json()
    text = body["choices"][0]["message"]["content"]
    return {"prompt": prompt, "response": text, "raw": body}


def load_prompts(input_file: Path, txt_mode: str) -> List[str]:
    if input_file.suffix.lower() == ".json":
        data = json.loads(input_file.read_text(encoding="utf-8"))
        if isinstance(data, list):
            if all(isinstance(x, str) for x in data):
                return data
            if all(isinstance(x, dict) and "prompt" in x for x in data):
                return [str(x["prompt"]) for x in data]
        raise ValueError("JSON input must be a list of strings or list of objects with key 'prompt'.")
    raw = input_file.read_text(encoding="utf-8")
    if txt_mode == "single":
        text = raw.strip()
        return [text] if text else []
    return [line.strip() for line in raw.splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference client for locally served vLLM model.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1", help="OpenAI-compatible endpoint.")
    parser.add_argument("--model", default=None, help="Model name. If omitted, discovered via /models.")
    parser.add_argument("--prompt", default=None, help="Single prompt to run.")
    parser.add_argument("--input-file", default=None, help="Optional txt/json file for batch prompts.")
    parser.add_argument("--output-file", default=None, help="Optional output json file for batch mode.")
    parser.add_argument(
        "--txt-mode",
        choices=["single", "lines"],
        default="single",
        help="How to read .txt input files: one full prompt ('single') or one prompt per non-empty line ('lines').",
    )
    parser.add_argument("--system-prompt", default="You are a helpful assistant.")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--timeout", type=float, default=120.0, help="Request timeout in seconds.")
    args = parser.parse_args()

    if not args.prompt and not args.input_file:
        raise ValueError("Provide either --prompt or --input-file.")

    model = args.model or discover_model(args.base_url, args.timeout)
    print(f"Using model: {model}")

    if args.prompt:
        result = infer_one(
            base_url=args.base_url,
            model=model,
            prompt=args.prompt,
            system_prompt=args.system_prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            timeout=args.timeout,
        )
        print(result["response"])
        return

    prompts = load_prompts(Path(args.input_file), args.txt_mode)
    results: List[Dict[str, Any]] = []
    for i, prompt in enumerate(prompts, start=1):
        result = infer_one(
            base_url=args.base_url,
            model=model,
            prompt=prompt,
            system_prompt=args.system_prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            timeout=args.timeout,
        )
        results.append({"index": i, "prompt": result["prompt"], "response": result["response"]})
        print(f"[{i}/{len(prompts)}] done")

    if args.output_file:
        output_path = Path(args.output_file)
        output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved outputs to {output_path}")
    else:
        print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
