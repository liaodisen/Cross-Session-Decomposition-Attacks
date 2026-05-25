"""Compose final answers from neutralized question batches using a research-focused LLM."""

import argparse
import json
import os
from typing import List, Dict, Any, Optional
from vllm import LLM, SamplingParams
from prompt import prompt_compose

# # HuggingFace cache location
# os.environ["HF_HOME"] = "/scratch/disen"

MODEL_NAME = "huihui-ai/Qwen2.5-14B-Instruct-abliterated-v2"


def build_llm(model: str, tensor_parallel_size: int) -> LLM:
    return LLM(
        model=model,
        dtype="bfloat16",
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=0.9,
        max_model_len=10000,
    )


def to_chat_prompt(tokenizer, messages: List[Dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def construct_summary_prompt(
    tokenizer,
    question_answer_pairs: Dict[str, Any],
) -> Optional[str]:
    """Construct the formatted chat prompt for a single batch of QA pairs."""
    qas = question_answer_pairs.get("answers", [])
    if not qas:
        return None
    qa_lines = []
    for qa in qas:
        q_text = qa.get("question", "").strip()
        a_text = qa.get("answer", "").strip()
        if q_text:
            qa_lines.append(f"Q: {q_text}\nA: {a_text}")
        else:
            qa_lines.append(f"A: {a_text}")
    qa_text = "\n".join(qa_lines)

    intent = question_answer_pairs.get("intent", "").strip() if question_answer_pairs else "Unknown Intent"

    system_prompt = "You are an expert in technical operations manual creation."
    user_prompt = prompt_compose.format(GOAL=intent, QA_PAIRS=qa_text)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return to_chat_prompt(tokenizer, messages)


def load_neutralized(path: str) -> Dict[str, List[Dict[str, str]]]:
    with open(path, "r") as f:
        return json.load(f)


def batch_generate(llm: LLM, prompts: List[str], sampling_params: SamplingParams) -> List[str]:
    outputs = llm.generate(prompts, sampling_params)
    return [out.outputs[0].text for out in outputs]


def domain_matches(entry_domain: str, domain_filter: str) -> bool:
    if domain_filter.strip().lower() in {"all", "*", ""}:
        return True
    return (entry_domain or "").strip().lower() == domain_filter.strip().lower()


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize neutralized question answers into final responses.")
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=2,
        help="Tensor parallel size for vLLM.",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Path(s) to neutralized answers JSON files.",
    )
    parser.add_argument(
        "--outputs",
        nargs="+",
        required=True,
        help="Output path(s) for summarized answers JSON; must match number of inputs.",
    )
    parser.add_argument(
        "--domain",
        default="all",
        help="Domain filter (e.g., chemistry, social, financial fraud, cybersecurity) or 'all'.",
    )
    args = parser.parse_args()

    if len(args.inputs) != len(args.outputs):
        raise SystemExit("Number of --inputs must equal number of --outputs")

    # Build LLM
    llm = build_llm(MODEL_NAME, args.tensor_parallel_size)
    tokenizer = llm.get_tokenizer()
    sampling_summary = SamplingParams(temperature=0.3, top_p=0.9, max_tokens=3000)

    for input_path, output_path in zip(args.inputs, args.outputs):
        data = load_neutralized(input_path)
        print(f"Loaded {len(data)} entries from {input_path}")

        # prepare prompts
        prompts: List[str] = []
        indices: List[Any] = []
        intents: List[str] = []
        domains: List[str] = []

        for index, qa_list in data.items():
            domain = qa_list.get("domain", "")
            if not domain_matches(domain, args.domain):
                continue
            prompt_str = construct_summary_prompt(tokenizer, qa_list)
            if prompt_str:
                prompts.append(prompt_str)
                indices.append(index)
                intents.append(qa_list.get("intent", "").strip())
                domains.append(domain)
        
        if not prompts:
            print(f"No valid prompts constructed for {input_path}. Skipping.")
            continue

        outputs = llm.generate(prompts, sampling_summary)

        # Collect Results
        results: List[Dict[str, Any]] = []
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text.strip()
            index_val = indices[i]
            
            results.append(
                {
                    "index": int(index_val) if str(index_val).isdigit() else index_val,
                    "domain": domains[i],
                    "intent": intents[i],
                    "summary_answer": generated_text,
                }
            )

        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"Saved {len(results)} summarized answers to {output_path}")


if __name__ == "__main__":
    main()
