#!/usr/bin/env python3
"""
Compare two or more judge runs side-by-side.

Usage:
  python3 compare.py ../results/judge_claude-sonnet-4-6_latest.json ../results/judge_Qwen3-5-9B-Base_latest.json
"""

import argparse
import json
import sys
from pathlib import Path


def load_judge(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def weighted(scores: dict) -> float:
    return (
        1.0 * scores["accuracy"]
        + 1.0 * scores["sources"]
        + 1.0 * scores["reasoning"]
        + 1.0 * scores["completeness"]
        + 0.5 * scores["language"]
    )


def fmt_row(cells, widths):
    return " | ".join(str(c).ljust(w) for c, w in zip(cells, widths))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+", help="judge_*.json files")
    p.add_argument("--per-subject", action="store_true")
    args = p.parse_args()

    runs = []
    for f in args.files:
        path = Path(f).resolve()
        data = load_judge(path)
        runs.append({"path": path, "data": data})

    # Header
    print("\n=== RabBench Comparison ===\n")
    labels = [r["data"]["meta"]["generation_model"] for r in runs]
    widths = [max(24, len(l) + 2) for l in labels]

    def row(label, getter):
        return fmt_row([label] + [getter(r) for r in runs], [30] + widths)

    print(row("Model", lambda r: r["data"]["meta"]["generation_model"]))
    print(row("Judge", lambda r: r["data"]["meta"]["judge_model"]))
    print(row("N scored", lambda r: r["data"]["summary"].get("n", 0)))
    print(row("Avg weighted /22.5", lambda r: r["data"]["summary"].get("avg_weighted_total")))
    print(row("Avg raw /25", lambda r: r["data"]["summary"].get("avg_raw_total")))
    print()

    # Per-dimension
    dims = ["accuracy", "sources", "reasoning", "completeness", "language"]
    print("Per-dimension averages (1-5):")
    for d in dims:
        print(row(f"  {d}", lambda r, d=d: r["data"]["summary"].get("avg_per_dim", {}).get(d, "-")))
    print()

    # Red flags
    print("Red flag counts:")
    all_flags: set[str] = set()
    for r in runs:
        all_flags.update(r["data"]["summary"].get("red_flag_counts", {}).keys())
    for f in sorted(all_flags):
        print(row(f"  {f}", lambda r, f=f: r["data"]["summary"].get("red_flag_counts", {}).get(f, 0)))
    print()

    if args.per_subject:
        all_subjects: set[str] = set()
        for r in runs:
            all_subjects.update(r["data"]["summary"].get("by_subject", {}).keys())
        print("Per-subject weighted avg:")
        for s in sorted(all_subjects):
            print(row(
                f"  {s}",
                lambda r, s=s: r["data"]["summary"].get("by_subject", {}).get(s, {}).get("avg_weighted", "-"),
            ))
        print()


if __name__ == "__main__":
    main()
