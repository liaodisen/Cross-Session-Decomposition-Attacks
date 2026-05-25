"""Analyze majority vote and judge agreement for pairwise eval result files."""

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


TIE_LABEL = "tie"
NO_MAJORITY_LABEL = "no_majority"


def slugify(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in value)
    safe = safe.strip("._-")
    return safe or "model"


def _sort_key(value: Any) -> Tuple[int, Any]:
    if isinstance(value, bool):
        return (0, int(value))
    if isinstance(value, int):
        return (0, value)
    if isinstance(value, float) and value.is_integer():
        return (0, int(value))
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit() or (stripped.startswith("-") and stripped[1:].isdigit()):
            return (0, int(stripped))
        return (1, stripped)
    return (2, str(value))


def _id_key(value: Any) -> str:
    return str(value)


def _round(value: Optional[float], digits: int = 6) -> Optional[float]:
    if value is None or not math.isfinite(value):
        return None
    return round(value, digits)


def normalize_winner(value: Any) -> str:
    winner = str(value or TIE_LABEL).strip().lower()
    if winner == "a":
        return "A"
    if winner == "b":
        return "B"
    return TIE_LABEL


def model_label(summary: Dict[str, Any], key: str, path: Path) -> str:
    model = summary.get(key)
    if not isinstance(model, dict):
        raise ValueError(f"{path} is missing {key}.label")
    label = str(model.get("label") or "").strip()
    if not label:
        raise ValueError(f"{path} is missing {key}.label")
    return label


def read_ready_summary(
    path: Path,
    wait: bool,
    wait_timeout: float,
    poll_interval: float,
) -> Dict[str, Any]:
    deadline = None if wait_timeout <= 0 else time.monotonic() + wait_timeout
    printed_wait = False
    last_error = ""

    while True:
        try:
            data = json.loads(path.read_text())
            if not isinstance(data, dict):
                raise ValueError("top-level JSON is not an object")

            comparisons = data.get("comparisons")
            if isinstance(comparisons, list) and comparisons:
                return data

            last_error = (
                f"{path} does not contain non-empty comparisons. "
                "Re-run blackbox/pair_eval.py with --save-comparisons."
            )
        except FileNotFoundError:
            last_error = f"{path} does not exist yet."
        except json.JSONDecodeError as exc:
            last_error = f"{path} is not valid JSON yet: {exc}"
        except ValueError as exc:
            last_error = f"{path}: {exc}"

        if not wait:
            raise ValueError(last_error)
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(last_error)
        if not printed_wait:
            print(f"Waiting for per-example comparisons in {path}", file=sys.stderr)
            printed_wait = True
        time.sleep(max(poll_interval, 0.1))


def discover_inputs(
    result_dir: Path,
    judge_labels: Sequence[str],
    label_a: str,
    label_b: str,
    wait: bool,
) -> List[Path]:
    paths: List[Path] = []
    a_slug = slugify(label_a)
    b_slug = slugify(label_b)

    for judge_label in judge_labels:
        judge_slug = slugify(judge_label)
        forward = result_dir / f"pair_eval_{judge_slug}_{a_slug}_vs_{b_slug}.json"
        reverse = result_dir / f"pair_eval_{judge_slug}_{b_slug}_vs_{a_slug}.json"

        if forward.exists():
            paths.append(forward)
        elif reverse.exists():
            paths.append(reverse)
        elif wait:
            paths.append(forward)
        else:
            raise FileNotFoundError(
                f"Could not find {forward} or {reverse} for judge {judge_label!r}"
            )

    return paths


def make_unique_judge_labels(raw_labels: Sequence[str], paths: Sequence[Path]) -> List[str]:
    counts = Counter(raw_labels)
    seen: Counter = Counter()
    unique: List[str] = []
    for label, path in zip(raw_labels, paths):
        seen[label] += 1
        if counts[label] == 1:
            unique.append(label)
        else:
            unique.append(f"{label}#{seen[label]}:{path.name}")
    return unique


def canonical_preference(
    winner: str,
    file_label_a: str,
    file_label_b: str,
    canonical_a: str,
    canonical_b: str,
    path: Path,
) -> str:
    if winner == TIE_LABEL:
        return TIE_LABEL

    if (file_label_a, file_label_b) == (canonical_a, canonical_b):
        return canonical_a if winner == "A" else canonical_b

    if (file_label_a, file_label_b) == (canonical_b, canonical_a):
        return canonical_b if winner == "A" else canonical_a

    raise ValueError(
        f"{path} compares {file_label_a!r} vs {file_label_b!r}, "
        f"not {canonical_a!r} vs {canonical_b!r}"
    )


def extract_preferences(
    summary: Dict[str, Any],
    path: Path,
    canonical_a: str,
    canonical_b: str,
    include_errors: bool,
) -> Tuple[Dict[str, str], Dict[str, Any], int]:
    file_label_a = model_label(summary, "model_a", path)
    file_label_b = model_label(summary, "model_b", path)
    comparisons = summary.get("comparisons")
    if not isinstance(comparisons, list) or not comparisons:
        raise ValueError(f"{path} does not contain non-empty comparisons")

    preferences: Dict[str, str] = {}
    id_values: Dict[str, Any] = {}
    skipped_errors = 0

    for comparison in comparisons:
        if not isinstance(comparison, dict):
            continue
        if comparison.get("error") and not include_errors:
            skipped_errors += 1
            continue
        item_id = comparison.get("id")
        key = _id_key(item_id)
        winner = normalize_winner(comparison.get("winner"))
        preferences[key] = canonical_preference(
            winner=winner,
            file_label_a=file_label_a,
            file_label_b=file_label_b,
            canonical_a=canonical_a,
            canonical_b=canonical_b,
            path=path,
        )
        id_values[key] = item_id

    return preferences, id_values, skipped_errors


def fleiss_kappa(rows: Sequence[Sequence[str]], categories: Sequence[str]) -> Optional[float]:
    if not rows:
        return None

    n_raters = len(rows[0])
    if n_raters < 2:
        return None

    category_totals: Counter = Counter()
    agreement_sum = 0.0

    for row in rows:
        if len(row) != n_raters:
            raise ValueError("Fleiss' kappa requires the same number of ratings per item")
        counts = Counter(row)
        category_totals.update(counts)
        agreement_sum += (
            sum(count * count for count in counts.values()) - n_raters
        ) / (n_raters * (n_raters - 1))

    observed = agreement_sum / len(rows)
    total_ratings = len(rows) * n_raters
    expected = sum((category_totals[category] / total_ratings) ** 2 for category in categories)

    denominator = 1.0 - expected
    if abs(denominator) < 1e-12:
        return 1.0 if abs(observed - 1.0) < 1e-12 else None
    return (observed - expected) / denominator


def krippendorff_alpha_nominal(rows: Sequence[Sequence[str]]) -> Optional[float]:
    if not rows:
        return None

    observed_disagreement = 0.0
    observed_pairs = 0
    category_totals: Counter = Counter()

    for row in rows:
        n_raters = len(row)
        if n_raters < 2:
            continue
        counts = Counter(row)
        category_totals.update(counts)
        observed_pairs += n_raters * (n_raters - 1)
        observed_disagreement += sum(count * (n_raters - count) for count in counts.values())

    if observed_pairs == 0:
        return None

    total_ratings = sum(category_totals.values())
    if total_ratings < 2:
        return None

    observed = observed_disagreement / observed_pairs
    expected = (
        sum(count * (total_ratings - count) for count in category_totals.values())
        / (total_ratings * (total_ratings - 1))
    )

    if abs(expected) < 1e-12:
        return 1.0 if abs(observed) < 1e-12 else None
    return 1.0 - (observed / expected)


def majority_vote(row: Sequence[str]) -> Tuple[str, int]:
    if not row:
        return NO_MAJORITY_LABEL, 0
    counts = Counter(row)
    max_count = max(counts.values())
    winners = [label for label, count in counts.items() if count == max_count]
    threshold = len(row) // 2 + 1
    if max_count >= threshold and len(winners) == 1:
        return winners[0], max_count
    return NO_MAJORITY_LABEL, max_count


def rate(count: int, total: int) -> float:
    return round(count / total, 6) if total else 0.0


def per_judge_summary(
    labels: Sequence[str],
    vectors: Dict[str, List[str]],
    canonical_a: str,
    canonical_b: str,
) -> Dict[str, Any]:
    summaries: Dict[str, Any] = {}
    for judge_label in labels:
        vector = vectors[judge_label]
        counts = Counter(vector)
        total = len(vector)
        non_ties = counts[canonical_a] + counts[canonical_b]
        summaries[judge_label] = {
            "counts": {
                canonical_a: counts[canonical_a],
                canonical_b: counts[canonical_b],
                TIE_LABEL: counts[TIE_LABEL],
            },
            "win_rate_all": {
                canonical_a: rate(counts[canonical_a], total),
                canonical_b: rate(counts[canonical_b], total),
            },
            "win_rate_non_tie": {
                canonical_a: rate(counts[canonical_a], non_ties),
                canonical_b: rate(counts[canonical_b], non_ties),
            },
        }
    return summaries


def analyze(
    paths: Sequence[Path],
    wait: bool,
    wait_timeout: float,
    poll_interval: float,
    canonical_a_arg: Optional[str],
    canonical_b_arg: Optional[str],
    include_errors: bool,
    include_vectors: bool,
) -> Dict[str, Any]:
    summaries = [
        read_ready_summary(
            path=path,
            wait=wait,
            wait_timeout=wait_timeout,
            poll_interval=poll_interval,
        )
        for path in paths
    ]

    canonical_a = canonical_a_arg or model_label(summaries[0], "model_a", paths[0])
    canonical_b = canonical_b_arg or model_label(summaries[0], "model_b", paths[0])

    raw_judge_labels = [
        str(summary.get("judge_model") or path.stem).strip() for summary, path in zip(summaries, paths)
    ]
    judge_labels = make_unique_judge_labels(raw_judge_labels, paths)

    preference_maps: Dict[str, Dict[str, str]] = {}
    id_values: Dict[str, Any] = {}
    skipped_error_counts: Dict[str, int] = {}

    for summary, path, judge_label in zip(summaries, paths, judge_labels):
        preferences, ids, skipped_errors = extract_preferences(
            summary=summary,
            path=path,
            canonical_a=canonical_a,
            canonical_b=canonical_b,
            include_errors=include_errors,
        )
        preference_maps[judge_label] = preferences
        id_values.update(ids)
        skipped_error_counts[judge_label] = skipped_errors

    if not preference_maps:
        raise ValueError("No judge files were loaded")

    common_ids = set.intersection(*(set(values) for values in preference_maps.values()))
    sorted_ids = sorted(common_ids, key=lambda key: _sort_key(id_values.get(key, key)))
    if not sorted_ids:
        raise ValueError("No shared comparison ids across judge files")

    vectors = {
        judge_label: [preference_maps[judge_label][key] for key in sorted_ids]
        for judge_label in judge_labels
    }
    rows = [[vectors[judge_label][idx] for judge_label in judge_labels] for idx in range(len(sorted_ids))]
    categories = [canonical_a, canonical_b, TIE_LABEL]

    majority_vector: List[str] = []
    majority_strengths: List[int] = []
    unanimous_count = 0
    for row in rows:
        label, count = majority_vote(row)
        majority_vector.append(label)
        majority_strengths.append(count)
        if count == len(row) and label != NO_MAJORITY_LABEL:
            unanimous_count += 1

    majority_counts = Counter(majority_vector)
    total_items = len(sorted_ids)
    majority_non_ties = majority_counts[canonical_a] + majority_counts[canonical_b]
    majority_consensus_count = total_items - majority_counts[NO_MAJORITY_LABEL]

    result: Dict[str, Any] = {
        "model_a": canonical_a,
        "model_b": canonical_b,
        "judge_files": [str(path) for path in paths],
        "judge_labels": judge_labels,
        "total_shared_items": total_items,
        "dropped_items_by_judge": {
            judge_label: len(preference_maps[judge_label]) - total_items
            for judge_label in judge_labels
        },
        "skipped_error_items_by_judge": skipped_error_counts,
        "per_judge": per_judge_summary(judge_labels, vectors, canonical_a, canonical_b),
        "majority_vote": {
            "counts": {
                canonical_a: majority_counts[canonical_a],
                canonical_b: majority_counts[canonical_b],
                TIE_LABEL: majority_counts[TIE_LABEL],
                NO_MAJORITY_LABEL: majority_counts[NO_MAJORITY_LABEL],
            },
            "win_rate_all": {
                canonical_a: rate(majority_counts[canonical_a], total_items),
                canonical_b: rate(majority_counts[canonical_b], total_items),
            },
            "win_rate_non_tie": {
                canonical_a: rate(majority_counts[canonical_a], majority_non_ties),
                canonical_b: rate(majority_counts[canonical_b], majority_non_ties),
            },
        },
        "agreement_metrics": {
            "fleiss_kappa": _round(fleiss_kappa(rows, categories)),
            "krippendorff_alpha_nominal": _round(krippendorff_alpha_nominal(rows)),
            "majority_consensus_rate": rate(majority_consensus_count, total_items),
            "unanimous_consensus_rate": rate(unanimous_count, total_items),
        },
    }

    if include_vectors:
        result["preference_vectors"] = {
            "ids": [id_values.get(key, key) for key in sorted_ids],
            "judges": judge_labels,
            "vectors": vectors,
            "majority_vector": majority_vector,
            "majority_strengths": majority_strengths,
        }

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract pairwise judge preference vectors, compute majority-vote "
            "win rates, and report Fleiss' kappa, Krippendorff's alpha, and "
            "majority consensus rate."
        )
    )
    parser.add_argument("inputs", nargs="*", help="Pair-eval JSON files to analyze.")
    parser.add_argument(
        "--input",
        dest="extra_inputs",
        action="append",
        default=[],
        help="Additional pair-eval JSON file. May be passed multiple times.",
    )
    parser.add_argument(
        "--result-dir",
        default="results",
        help="Directory used with --judge-labels/--model-a/--model-b discovery.",
    )
    parser.add_argument(
        "--judge-labels",
        nargs="+",
        help="Judge labels to discover in results/pair_eval_<judge>_<A>_vs_<B>.json.",
    )
    parser.add_argument("--model-a", help="Canonical model A label.")
    parser.add_argument("--model-b", help="Canonical model B label.")
    parser.add_argument(
        "--include-errors",
        action="store_true",
        help="Keep comparison rows that have a non-null error field.",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Poll until each input file has non-empty per-example comparisons.",
    )
    parser.add_argument(
        "--wait-timeout",
        type=float,
        default=3600.0,
        help="Seconds to wait before failing. Use 0 to wait indefinitely.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=30.0,
        help="Seconds between --wait readiness checks.",
    )
    parser.add_argument(
        "--no-vectors",
        action="store_true",
        help="Omit per-example preference vectors from the output JSON.",
    )
    parser.add_argument("--output", help="Optional output JSON path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = [Path(value) for value in args.inputs + args.extra_inputs]

    if not paths:
        if not args.judge_labels or not args.model_a or not args.model_b:
            raise SystemExit(
                "Provide input files, or provide --judge-labels with --model-a and --model-b."
            )
        paths = discover_inputs(
            result_dir=Path(args.result_dir),
            judge_labels=args.judge_labels,
            label_a=args.model_a,
            label_b=args.model_b,
            wait=args.wait,
        )

    try:
        result = analyze(
            paths=paths,
            wait=args.wait,
            wait_timeout=args.wait_timeout,
            poll_interval=args.poll_interval,
            canonical_a_arg=args.model_a,
            canonical_b_arg=args.model_b,
            include_errors=args.include_errors,
            include_vectors=not args.no_vectors,
        )
    except (FileNotFoundError, TimeoutError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from None

    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text)
        print(f"Saved pairwise judge agreement summary to {output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
