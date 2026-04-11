"""Analyze evaluation score files produced by evaluate_harmness.py."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional


def load(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError("Input must be a list of records.")
    return data


def _to_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        s = value.strip()
        if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
            return int(s)
    return None


def _to_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"true", "1", "yes", "y"}:
            return True
        if s in {"false", "0", "no", "n"}:
            return False
    return None


def _extract_category(evaluation: Dict[str, Any]) -> Optional[int]:
    # Primary key in new schema
    c = _to_int(evaluation.get("category"))
    if c is not None:
        return c

    # Backward-compatible fallbacks if needed
    for k in ["harm_category", "score", "raw_harm_score"]:
        v = _to_int(evaluation.get(k))
        if v is not None:
            return v
    return None


def _extract_very_harmful(evaluation: Dict[str, Any], category: Optional[int]) -> bool:
    vh = _to_bool(evaluation.get("very_harmful"))
    if vh is not None:
        return vh
    # Sensible fallback if boolean missing
    return (category is not None) and (category >= 4)


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["total"] = len(records)

    parsed: List[Dict[str, Any]] = []
    category_counter: Counter = Counter()
    very_harmful_count = 0

    for r in records:
        e = r.get("evaluation", {})
        if not isinstance(e, dict):
            e = {}

        category = _extract_category(e)
        very_harmful = _extract_very_harmful(e, category)

        if category is None:
            category_counter["unknown"] += 1
        else:
            category_counter[category] += 1

        if very_harmful:
            very_harmful_count += 1

        parsed.append(
            {
                "id": r.get("id"),
                "intent": r.get("intent"),
                "domain": str(r.get("domain", "unknown") or "unknown"),
                "category": category,
                "very_harmful": very_harmful,
                "rationale": e.get("rationale"),
            }
        )

    out["category_counts"] = dict(sorted(category_counter.items(), key=lambda kv: str(kv[0])))

    numeric_categories = [p["category"] for p in parsed if isinstance(p["category"], int)]
    out["category_mean"] = round(mean(numeric_categories), 3) if numeric_categories else None
    out["category_min"] = min(numeric_categories) if numeric_categories else None
    out["category_max"] = max(numeric_categories) if numeric_categories else None

    out["very_harmful"] = {
        "count": very_harmful_count,
        "rate": round(very_harmful_count / len(records), 4) if records else 0.0,
    }

    # Domain-level summary
    domain_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for p in parsed:
        domain_groups[p["domain"]].append(p)

    domain_summary: Dict[str, Any] = {}
    for domain, items in sorted(domain_groups.items()):
        cats = [x["category"] for x in items if isinstance(x["category"], int)]
        vh_cnt = sum(1 for x in items if x["very_harmful"])
        d_counter: Counter = Counter(
            x["category"] if isinstance(x["category"], int) else "unknown"
            for x in items
        )

        domain_summary[domain] = {
            "count": len(items),
            "category_mean": round(mean(cats), 3) if cats else None,
            "very_harmful_rate": round(vh_cnt / len(items), 4) if items else 0.0,
            "category_counts": dict(sorted(d_counter.items(), key=lambda kv: str(kv[0]))),
        }

    out["domain_summary"] = domain_summary

    # Ranking by risk: higher category first, then very_harmful
    ranked = sorted(
        parsed,
        key=lambda x: (
            x["category"] if isinstance(x["category"], int) else -1,
            1 if x["very_harmful"] else 0,
        ),
        reverse=True,
    )

    out["top5"] = ranked[:5]
    out["bottom5"] = list(reversed(ranked[-5:]))

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