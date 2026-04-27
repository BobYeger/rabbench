#!/usr/bin/env python3
"""
Prepare the v2 mixed training set: 80% domain + 20% general replay.

Reads the existing domain SFT files (do NOT regenerate), downloads a slice of a
general instruction dataset (UltraChat-200k by default), normalizes both to the
{"messages": [...]} format, mixes 80/20, shuffles, and writes a single training
file plus a small mixed validation file.

The replay slice prevents catastrophic forgetting: in v1 the model lost 12.6pp
on the factual probe after 3 epochs of domain-only SFT. Mixed data + low LR is
the standard fix (e.g. PaLM 2, LIMA, OpenAssistant).

Usage:
  /home/aigroup/training-env/bin/python train/prepare_data.py
  /home/aigroup/training-env/bin/python train/prepare_data.py --replay-ratio 0.2
  /home/aigroup/training-env/bin/python train/prepare_data.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "train" / "configs" / "v2_mixed.json"


def load_config(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_jsonl_messages(path: Path) -> list[dict]:
    out: list[dict] = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  skip line {i}: {e}", file=sys.stderr)
                continue
            msgs = rec.get("messages")
            if msgs and isinstance(msgs, list):
                out.append({"messages": msgs, "source": rec.get("source", "domain"), "tier": rec.get("tier", "domain")})
    return out


def normalize_replay_messages(record: dict, max_turns: int) -> dict | None:
    """
    UltraChat-200k records ship as {"messages": [{"role": "user"/"assistant", "content": ...}, ...]}.
    Trim to <= max_turns turns and require the last turn to be assistant.
    """
    msgs = record.get("messages") or []
    msgs = [m for m in msgs if m.get("role") in ("system", "user", "assistant") and m.get("content")]
    if len(msgs) < 2:
        return None
    if msgs[-1]["role"] != "assistant":
        msgs = msgs[:-1] if len(msgs) >= 2 and msgs[-2]["role"] == "assistant" else msgs
    if msgs and msgs[-1]["role"] != "assistant":
        return None
    if len(msgs) > max_turns:
        msgs = msgs[:max_turns]
        if msgs[-1]["role"] != "assistant":
            msgs = msgs[:-1]
    if len(msgs) < 2 or msgs[-1]["role"] != "assistant":
        return None
    return {"messages": msgs, "source": "replay:ultrachat", "tier": "replay"}


def load_replay(name: str, split: str, n: int, max_turns: int, seed: int) -> list[dict]:
    """
    Stream the replay dataset from HuggingFace; pull until we have n usable
    records. Streaming avoids downloading the full 200k-row parquet.
    """
    from datasets import load_dataset
    print(f"→ streaming {name} [{split}] for {n} samples (max_turns={max_turns})", file=sys.stderr)

    ds = load_dataset(name, split=split, streaming=True).shuffle(seed=seed, buffer_size=10_000)
    out: list[dict] = []
    seen = 0
    for rec in ds:
        seen += 1
        norm = normalize_replay_messages(rec, max_turns=max_turns)
        if norm is not None:
            out.append(norm)
            if len(out) >= n:
                break
        if seen > n * 20:
            print(f"  warn: scanned {seen} but only {len(out)} usable; stopping early", file=sys.stderr)
            break
    print(f"  kept {len(out)} / {seen} scanned", file=sys.stderr)
    return out


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def stats(records: list[dict]) -> dict:
    by_tier: dict[str, int] = {}
    for r in records:
        by_tier[r.get("tier", "?")] = by_tier.get(r.get("tier", "?"), 0) + 1
    return {"total": len(records), "by_tier": by_tier}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--replay-ratio", type=float, default=None,
                   help="Override config: fraction of total mix that is replay (0.20 default)")
    p.add_argument("--replay-dataset", default=None, help="Override replay dataset name")
    p.add_argument("--replay-split", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--dry-run", action="store_true", help="Compute and report sizes; do not write files")
    args = p.parse_args()

    cfg = load_config(Path(args.config))
    replay_ratio = args.replay_ratio if args.replay_ratio is not None else cfg["replay_ratio"]
    replay_dataset = args.replay_dataset or cfg["replay_dataset"]
    replay_split = args.replay_split or cfg["replay_split"]
    max_turns = cfg.get("replay_max_turns", 4)
    seed = args.seed if args.seed is not None else cfg["seed"]

    domain_train = load_jsonl_messages(Path(cfg["domain_train_file"]))
    domain_val = load_jsonl_messages(Path(cfg["domain_val_file"]))
    print(f"domain_train: {len(domain_train)} | domain_val: {len(domain_val)}", file=sys.stderr)

    if not (0 < replay_ratio < 1):
        raise ValueError(f"replay_ratio must be in (0, 1), got {replay_ratio}")

    n_replay_train = round(len(domain_train) * replay_ratio / (1 - replay_ratio))
    n_replay_val = round(len(domain_val) * replay_ratio / (1 - replay_ratio))

    print(
        f"target mix: {len(domain_train)} domain + {n_replay_train} replay "
        f"= {len(domain_train) + n_replay_train} total ({replay_ratio:.0%} replay)",
        file=sys.stderr,
    )

    if args.dry_run:
        print("--dry-run: not downloading or writing", file=sys.stderr)
        return

    replay_train = load_replay(replay_dataset, replay_split, n_replay_train, max_turns, seed)
    replay_val_split = "test_sft" if "ultrachat" in replay_dataset else replay_split
    try:
        replay_val = load_replay(replay_dataset, replay_val_split, n_replay_val, max_turns, seed + 1)
    except Exception as e:
        print(f"  replay val split {replay_val_split} unavailable ({e}); reusing train slice", file=sys.stderr)
        replay_val = load_replay(replay_dataset, replay_split, n_replay_val, max_turns, seed + 1)

    rng = random.Random(seed)
    mixed_train = domain_train + replay_train
    rng.shuffle(mixed_train)
    mixed_val = domain_val + replay_val
    rng.shuffle(mixed_val)

    out_train = Path(cfg["mixed_train_file"])
    out_val = Path(cfg["mixed_val_file"])
    write_jsonl(mixed_train, out_train)
    write_jsonl(mixed_val, out_val)

    summary = {
        "domain_train": len(domain_train),
        "domain_val": len(domain_val),
        "replay_train": len(replay_train),
        "replay_val": len(replay_val),
        "mixed_train": stats(mixed_train),
        "mixed_val": stats(mixed_val),
        "replay_dataset": replay_dataset,
        "replay_split": replay_split,
        "replay_ratio": replay_ratio,
        "max_turns": max_turns,
        "seed": seed,
        "out_train": str(out_train),
        "out_val": str(out_val),
    }
    out_stats = out_train.parent / "mixed_stats.json"
    with open(out_stats, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n✓ wrote {out_train}  ({len(mixed_train)} records)")
    print(f"✓ wrote {out_val}    ({len(mixed_val)} records)")
    print(f"✓ stats {out_stats}")
    actual = len(replay_train) / max(1, len(mixed_train))
    print(f"  actual replay fraction: {actual:.1%}")


if __name__ == "__main__":
    main()
