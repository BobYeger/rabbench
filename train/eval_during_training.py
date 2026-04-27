#!/usr/bin/env python3
"""
Mid-training factual probe — call between save_steps to verify a checkpoint
hasn't regressed on facts the base model already knew.

Two modes:

  1) Against an already-running vLLM server (the recommended path):
       python3 train/eval_during_training.py \\
           --base-url http://localhost:8000/v1 --model rabbench-sft-v2-step500

  2) From a checkpoint dir (merges LoRA into a temp bf16 model and prints
     the vLLM command to run; doesn't launch vLLM itself — hardware-specific
     flags differ per box):
       python3 train/eval_during_training.py \\
           --checkpoint /home/aigroup/training/rabbench-sft-v2/checkpoint-500 \\
           --print-serve-cmd

Always compares against a baseline probe result file (default: the
Qwen3.5-9B-Base probe under results/). Exits 0 on PASS, 2 on FAIL
(>= --threshold-pp drop), 1 on operational error.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = REPO_ROOT / "bench"
DEFAULT_BASELINE = REPO_ROOT / "results" / "probe_Qwen3.5-9B_latest.json"


def load_summary(path: Path) -> dict:
    with open(path) as f:
        data = json.load(f)
    return data["summary"]


def find_latest_probe_for(model_name: str) -> Path | None:
    """Find the most recent probe file for this model name."""
    results_dir = REPO_ROOT / "results"
    candidates = sorted(results_dir.glob(f"probe_*{model_name}*.json"), reverse=True)
    return candidates[0] if candidates else None


def run_probe(model: str, base_url: str, pilot: int | None) -> Path:
    """Invoke bench/probe.py against the given vLLM endpoint; return the result path."""
    cmd = [
        sys.executable, "probe.py",
        "--model", model,
        "--provider", "openai",
        "--base-url", base_url,
    ]
    if pilot is not None:
        cmd += ["--pilot", str(pilot)]
    print(f"→ {' '.join(cmd)}", file=sys.stderr)
    proc = subprocess.run(cmd, cwd=str(BENCH_DIR), check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"probe.py exited {proc.returncode}")
    out = find_latest_probe_for(model)
    if out is None:
        raise RuntimeError("probe ran but no result file was produced")
    return out


def merge_checkpoint(checkpoint: Path, base_model: Path, out_dir: Path) -> Path:
    """Merge a LoRA adapter checkpoint into a bf16 directory ready for vLLM."""
    print(f"→ merging {checkpoint} + {base_model} -> {out_dir}", file=sys.stderr)
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(str(base_model), torch_dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(base, str(checkpoint)).merge_and_unload()
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out_dir))
    tok = AutoTokenizer.from_pretrained(str(checkpoint))
    tok.save_pretrained(str(out_dir))
    base_tok = base_model / "tokenizer_config.json"
    if base_tok.exists():
        shutil.copy(str(base_tok), str(out_dir / "tokenizer_config.json"))
    return out_dir


def verdict(probe_acc: float, baseline_acc: float, threshold_pp: float) -> tuple[str, int]:
    delta_pp = (probe_acc - baseline_acc) * 100
    if delta_pp <= -threshold_pp:
        return f"FAIL  (Δ {delta_pp:+.1f}pp ≤ -{threshold_pp:.1f}pp)", 2
    if delta_pp < 0:
        return f"WARN  (Δ {delta_pp:+.1f}pp, within tolerance)", 0
    return f"PASS  (Δ {delta_pp:+.1f}pp)", 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", help="Model name as known to the vLLM server")
    p.add_argument("--base-url", default="http://localhost:8000/v1",
                   help="vLLM OpenAI-compatible endpoint")
    p.add_argument("--baseline", default=str(DEFAULT_BASELINE),
                   help="Path to baseline probe JSON (default: Qwen3.5-9B-Base)")
    p.add_argument("--threshold-pp", type=float, default=3.0,
                   help="Forgetting threshold; drop >= this is FAIL")
    p.add_argument("--pilot", type=int, default=None,
                   help="Run only the first N questions for a fast sanity check")

    p.add_argument("--checkpoint", help="LoRA adapter dir to merge before serving")
    p.add_argument("--base-model", default="/home/aigroup/models/Qwen3.5-9B-Base",
                   help="Base model dir (only used with --checkpoint)")
    p.add_argument("--merged-dir", help="Where to write merged bf16 (only used with --checkpoint)")
    p.add_argument("--print-serve-cmd", action="store_true",
                   help="Merge checkpoint and print vLLM serve command, then exit")

    args = p.parse_args()

    if args.checkpoint:
        ckpt = Path(args.checkpoint)
        if not ckpt.exists():
            print(f"checkpoint not found: {ckpt}", file=sys.stderr)
            return 1
        merged_dir = Path(args.merged_dir) if args.merged_dir else ckpt.parent / f"{ckpt.name}-merged"
        if not (merged_dir / "config.json").exists():
            merge_checkpoint(ckpt, Path(args.base_model), merged_dir)
        else:
            print(f"→ reusing existing merged dir {merged_dir}", file=sys.stderr)
        print(f"\nMerged model ready: {merged_dir}")
        if args.print_serve_cmd:
            print("\n# Suggested vLLM serve command (sm_121 / GB10):")
            print(
                f"docker run --rm -p 8000:8000 --gpus all "
                f"-v {merged_dir}:{merged_dir} "
                f"vllm-openai:qwen3_5-hardened "
                f"--model {merged_dir} --enforce-eager "
                f"--served-model-name {merged_dir.name}"
            )
            return 0
        if not args.model:
            args.model = merged_dir.name

    if not args.model:
        print("--model is required (or pass --checkpoint --print-serve-cmd)", file=sys.stderr)
        return 1

    baseline_path = Path(args.baseline)
    if not baseline_path.exists():
        print(f"baseline missing: {baseline_path}", file=sys.stderr)
        return 1
    baseline_acc = load_summary(baseline_path)["accuracy"]
    print(f"→ baseline {baseline_path.name}: {baseline_acc:.1%}", file=sys.stderr)

    try:
        result_path = run_probe(args.model, args.base_url, args.pilot)
    except Exception as e:
        print(f"probe failed: {e}", file=sys.stderr)
        return 1

    summary = load_summary(result_path)
    probe_acc = summary["accuracy"]
    msg, code = verdict(probe_acc, baseline_acc, args.threshold_pp)

    print(f"\nbaseline:  {baseline_acc:.1%}")
    print(f"checkpoint: {probe_acc:.1%}  ({summary['n_correct']}/{summary['n_scored']})")
    print(f"verdict:   {msg}")
    print(f"result:    {result_path}")

    by_subj = summary.get("by_subject", {})
    base_by_subj = load_summary(baseline_path).get("by_subject", {})
    drops = []
    for subj, s in by_subj.items():
        b = base_by_subj.get(subj)
        if not b:
            continue
        d = (s["accuracy"] - b["accuracy"]) * 100
        if d < -args.threshold_pp:
            drops.append((subj, d, b["accuracy"], s["accuracy"]))
    if drops:
        drops.sort(key=lambda x: x[1])
        print(f"\nlargest per-subject regressions (>{args.threshold_pp:.1f}pp):")
        for subj, d, ba, sa in drops[:5]:
            print(f"  {subj:24s}  {ba:.1%} → {sa:.1%}  (Δ {d:+.1f}pp)")

    return code


if __name__ == "__main__":
    sys.exit(main())
