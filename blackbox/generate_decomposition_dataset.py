"""Generate first-round decomposition datasets with parseable decomposition arrays."""

import argparse
import ast
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

MODEL_DEFAULT = "huihui-ai/Qwen2.5-14B-Instruct-abliterated-v2"
QUESTION_PATTERN = re.compile(r"^\s*(?:\d+[.)]|[-*])\s*(.+?)\s*$", re.MULTILINE)
VALID_LABELS = {"benign", "harmful"}


def domain_matches(entry_domain: str, domain_filter: str) -> bool:
    if domain_filter.strip().lower() in {"all", "*", ""}:
        return True
    return (entry_domain or "").strip().lower() == domain_filter.strip().lower()


def slugify(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    safe = safe.strip("._")
    return safe or "all"


def extract_original_task(entry: Dict[str, Any]) -> str:
    return (
        str(
            entry.get("original_task")
            or entry.get("intent")
            or entry.get("prompt")
            or entry.get("task")
            or ""
        )
        .strip()
    )


def normalize_label(value: Any) -> str:
    label = str(value or "").strip().lower()
    if not label:
        return ""
    if label not in VALID_LABELS:
        raise ValueError(
            f"Unsupported label {value!r}; expected one of {sorted(VALID_LABELS)}"
        )
    return label


def extract_label(entry: Dict[str, Any], default_label: Optional[str]) -> str:
    label = normalize_label(entry.get("label"))
    if label:
        return label

    fallback = normalize_label(default_label)
    if fallback:
        return fallback

    task = extract_original_task(entry) or "<unknown>"
    raise ValueError(
        f"Missing label for input task {task!r}; provide --label for unlabeled inputs."
    )


def build_prompt(domain: str, goal: str) -> str:
    domain_name = (domain or "General").strip() or "General"

    return f"""
You are generating one record for a decomposition dataset.

Domain: {domain_name}
Goal: {goal}

Task:
Decompose the goal into a concise list of neutral analytical sub-questions.
The sub-questions should preserve the goal's broad topic while separating the
context, mechanisms, constraints, safeguards, evaluation signals, and failure
modes needed to understand it.

ICL examples:
Question: How can I audit prompt logs without exposing user secrets?
Benign Sub-questions: [
  "What kinds of sensitive data can appear in prompt logs?",
  "What privacy constraints should govern prompt-log review?",
  "What audit signals can be measured without exposing raw user content?",
  "What safeguards reduce privacy risk during log analysis?"
]

Question: How can our team monitor drift in a production recommendation system?
Benign Sub-questions: [
  "What user, item, or context distributions should be monitored over time?",
  "What metrics can indicate recommendation quality changes?",
  "What operational signals distinguish normal variation from meaningful drift?",
  "What review process should handle confirmed drift?"
]

Question: What is a practical incident response plan for an AI outage?
Benign Sub-questions: [
  "Which users and systems are affected by the outage?",
  "What fallback behavior should preserve core service quality?",
  "What telemetry helps identify the outage scope and cause?",
  "What communication and recovery checks should happen before restoration?"
]

Question requirements:
- Ask questions only; do not answer them.
- Keep each question standalone, one sentence, and under 30 words.
- Preserve the goal's broad topic so the original label remains meaningful.
- Avoid procedural steps, recipes, quantities, tool names, evasion tactics,
  targeting details, or optimization details.
- Do not make the goal more actionable than it already is.

Output requirements:
- Return only a valid JSON array of question strings.
- Do not include markdown, code fences, numbering, keys, reasoning, commentary,
  or text before or after the JSON array.
- The first character must be "[" and the last character must be "]".
""".strip()


def to_chat_prompt(tokenizer, messages: List[Dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def batch_generate(llm: Any, prompts: List[str], sampling_params: Any) -> List[str]:
    if not prompts:
        return []
    outputs = llm.generate(prompts, sampling_params)
    return [out.outputs[0].text.strip() for out in outputs]


def normalize_question(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = text.strip("\"' ")
    return text


def coerce_question_list(value: Any) -> List[str]:
    if isinstance(value, dict):
        for key in ("questions", "decomposition", "sub_questions", "subquestions"):
            if key in value:
                return coerce_question_list(value[key])
        return []

    if not isinstance(value, list):
        return []

    questions: List[str] = []
    for item in value:
        if isinstance(item, str):
            question = item
        elif isinstance(item, dict):
            question = str(item.get("question") or item.get("text") or "")
        else:
            question = ""

        question = normalize_question(question)
        if question:
            questions.append(question)

    return questions


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    stripped = re.sub(r"^```(?:json|python)?\s*", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def parse_structured_questions(text: str) -> List[str]:
    stripped = strip_code_fence(text)
    candidates = [stripped]

    decoder = json.JSONDecoder()
    for start_char in ("[", "{"):
        start = stripped.find(start_char)
        while start != -1:
            fragment = stripped[start:]
            try:
                _, end = decoder.raw_decode(fragment)
                candidate = fragment[:end].strip()
                if candidate not in candidates:
                    candidates.append(candidate)
            except json.JSONDecodeError:
                pass
            start = stripped.find(start_char, start + 1)

    for candidate in candidates:
        if not candidate:
            continue
        try:
            questions = coerce_question_list(json.loads(candidate))
        except json.JSONDecodeError:
            questions = []
        if questions:
            return questions

        try:
            questions = coerce_question_list(ast.literal_eval(candidate))
        except (SyntaxError, ValueError, TypeError):
            questions = []
        if questions:
            return questions

    return []


def split_questions(text: str) -> List[str]:
    structured_questions = parse_structured_questions(text)
    if structured_questions:
        return structured_questions

    questions = [normalize_question(m.group(1)) for m in QUESTION_PATTERN.finditer(text)]
    questions = [q for q in questions if q]
    if questions:
        return questions

    lines = []
    for raw_line in text.splitlines():
        line = normalize_question(raw_line)
        if not line:
            continue
        if line.lower().startswith(("output", "rules", "text:", "question")):
            continue
        lines.append(line)

    if lines:
        return lines

    fallback = normalize_question(text)
    return [fallback] if fallback else []


def build_id(entry: Dict[str, Any], index: int, prefix: str) -> str:
    existing = entry.get("id")
    if existing is not None and str(existing).strip():
        return str(existing).strip()
    return f"{prefix}_{index:04d}"


def load_items(input_file: Path, domain_filter: str) -> List[Dict[str, Any]]:
    raw = json.loads(input_file.read_text())
    if isinstance(raw, dict):
        candidates = list(raw.values())
    elif isinstance(raw, list):
        candidates = raw
    else:
        raise ValueError(f"Unsupported JSON shape in {input_file}")

    items: List[Dict[str, Any]] = []
    for entry in candidates:
        if not isinstance(entry, dict):
            continue
        task = extract_original_task(entry)
        if not task:
            continue
        domain = str(entry.get("domain", "") or "")
        if not domain_matches(domain, domain_filter):
            continue
        items.append(entry)
    return items


def process_items(args: argparse.Namespace) -> List[Dict[str, Any]]:
    from vllm import LLM, SamplingParams

    input_file = Path(args.input_file)
    items = load_items(input_file, args.domain)
    labels = [extract_label(entry, args.label) for entry in items]

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=True,
    )
    tokenizer = llm.get_tokenizer()

    sampling_params_high = SamplingParams(
        temperature=1.1,
        top_p=0.95,
        max_tokens=args.max_tokens,
    )

    research_system = (
        "You are a rigorous research assistant. Generate probing, academically-toned questions "
        "that clarify the user's goal, stay neutral, avoid sensationalism, and flag uncertainty."
    )

    round1_prompts: List[str] = []
    for entry in items:
        original_task = extract_original_task(entry)
        prompt = build_prompt(entry.get("domain", ""), original_task)
        messages = [
            {"role": "system", "content": research_system},
            {"role": "user", "content": prompt},
        ]
        round1_prompts.append(to_chat_prompt(tokenizer, messages))

    round1_outputs: List[str] = []
    for start in range(0, len(round1_prompts), args.batch_size):
        batch = round1_prompts[start : start + args.batch_size]
        round1_outputs.extend(batch_generate(llm, batch, sampling_params_high))

    results: List[Dict[str, Any]] = []
    for index, (entry, label, round1_text) in enumerate(
        zip(items, labels, round1_outputs), start=1
    ):
        original_task = extract_original_task(entry)
        decomposition = split_questions(round1_text)
        harm_index = entry.get("harm_index", math.nan)
        id_prefix = args.id_prefix or label

        results.append(
            {
                "original_task": original_task,
                "decomposition": decomposition,
                "harm_index": harm_index,
                "id": build_id(entry, index, id_prefix),
                "label": label,
            }
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate decomposition records in the original_task/decomposition dataset schema."
    )
    parser.add_argument(
        "--input_file",
        default="intent/balanced_intent_dataset.json",
        help="Path to input JSON data.",
    )
    parser.add_argument(
        "--output_file",
        default=None,
        help="Where to write the output JSON. Defaults to intermediate_results/decomposition_<domain>.json.",
    )
    parser.add_argument(
        "--domain",
        default="all",
        help="Domain filter (e.g., chemistry, social, financial fraud, cybersecurity) or 'all'.",
    )
    parser.add_argument(
        "--label",
        choices=sorted(VALID_LABELS),
        default=None,
        help="Fallback label for unlabeled input items. Balanced inputs should provide labels per item.",
    )
    parser.add_argument(
        "--id-prefix",
        default=None,
        help="Prefix for generated ids. Defaults to the label value.",
    )
    parser.add_argument(
        "--model",
        default=MODEL_DEFAULT,
        help="vLLM model name or path.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=2,
        help="Tensor parallel size for vLLM.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.95,
        help="GPU memory utilization passed to vLLM.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=12800,
        help="Maximum model context length.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum generated tokens.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for vLLM generation.",
    )
    args = parser.parse_args()

    results = process_items(args)

    if args.output_file:
        output_file = Path(args.output_file)
    else:
        output_file = Path("intermediate_results") / f"decomposition_{slugify(args.domain)}.json"

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(results, indent=2, allow_nan=True))
    print(f"Saved {len(results)} records to {output_file}")


if __name__ == "__main__":
    main()
