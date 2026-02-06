"""Analyze evaluation score files produced by evaluate_harmness.py."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List


def load(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError("Input must be a list of records.")
    return data


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    evals = [r.get("evaluation", {}) for r in records]
    out: Dict[str, Any] = {}

    # Counts
    out["total"] = len(evals)

    # Detect numeric axes dynamically (exclude raw/harm_level)
    candidate_keys = set().union(*(e.keys() for e in evals))
    numeric_axes = []
    for k in sorted(candidate_keys):
        if k in {"raw_harm_score", "harm_level", "rationale"}:
            continue
        values = [e.get(k) for e in evals]
        if all(isinstance(v, (int, float)) for v in values if v is not None):
            numeric_axes.append(k)

    out["axes"] = numeric_axes
    out["axis_mean"] = {axis: round(mean([e.get(axis, 0) for e in evals]), 3) for axis in numeric_axes}

    # Raw score stats (fallback to summed axes if raw_harm_score missing)
    if any("raw_harm_score" in e for e in evals):
        raw_scores = [e.get("raw_harm_score", 0) for e in evals]
    else:
        raw_scores = [sum(e.get(a, 0) for a in numeric_axes) for e in evals]

    out["raw_score"] = {
        "mean": round(mean(raw_scores), 3),
        "min": min(raw_scores),
        "max": max(raw_scores),
    }

    # Harm level distribution if present
    if any("harm_level" in e for e in evals):
        harms = Counter(e.get("harm_level", "unknown") for e in evals)
        out["harm_level_counts"] = dict(sorted(harms.items()))

    # Top/bottom by raw score
    sorted_records = sorted(
        records,
        key=lambda r: r.get("evaluation", {}).get("raw_harm_score", sum(r.get("evaluation", {}).get(a, 0) for a in numeric_axes)),
        reverse=True,
    )
    out["top5"] = [
        {
            "id": r.get("id"),
            "intent": r.get("intent"),
            "raw_harm_score": r.get("evaluation", {}).get("raw_harm_score", None),
            "harm_level": r.get("evaluation", {}).get("harm_level", None),
        }
        for r in sorted_records[:5]
    ]
    out["bottom5"] = [
        {
            "id": r.get("id"),
            "intent": r.get("intent"),
            "raw_harm_score": r.get("evaluation", {}).get("raw_harm_score", None),
            "harm_level": r.get("evaluation", {}).get("harm_level", None),
        }
        for r in sorted_records[-5:]
    ]

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze harm evaluation scores.")
    parser.add_argument("--input", required=True, help="Path to eval JSON (list of records).")
    parser.add_argument(
        "--output",
        help="Optional path to save summary JSON. If omitted, prints to stdout.",
    )
    args = parser.parse_args()

    records = load(Path(args.input))
    summary = summarize(records)

    if args.output:
        Path(args.output).write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"Summary written to {args.output}")
    else:
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
