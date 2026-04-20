#!/usr/bin/env python3
"""
Merge a LoRA adapter back into the base model (bf16).

train/sft.py already writes a merged/ directory at the end. Use this script when
you want to merge a specific checkpoint, or re-merge after editing the adapter.

Usage:
  /home/aigroup/training-env/bin/python train/merge.py \\
      --base /home/aigroup/models/Qwen3.5-9B-Base \\
      --adapter /home/aigroup/training/rabbench-sft/adapter \\
      --output /home/aigroup/training/rabbench-sft/merged
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True, help="Path to the base model")
    p.add_argument("--adapter", required=True, help="Path to the trained LoRA adapter")
    p.add_argument("--output", required=True, help="Where to write the merged bf16 model")
    p.add_argument(
        "--backend",
        choices=["unsloth", "peft"],
        default="unsloth",
        help="Merge implementation (default: unsloth, which emits sharded safetensors)",
    )
    args = p.parse_args()

    base = Path(args.base).resolve()
    adapter = Path(args.adapter).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    print(f"Base:     {base}")
    print(f"Adapter:  {adapter}")
    print(f"Output:   {output}")
    print(f"Backend:  {args.backend}")

    import torch

    if args.backend == "unsloth":
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(base),
            max_seq_length=4096,
            load_in_4bit=False,
            load_in_16bit=True,
            dtype=torch.bfloat16,
        )
        model.load_adapter(str(adapter))
        model.save_pretrained_merged(str(output), tokenizer, save_method="merged_16bit")

    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        tokenizer = AutoTokenizer.from_pretrained(str(base), trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            str(base),
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
        peft_model = PeftModel.from_pretrained(base_model, str(adapter))
        merged = peft_model.merge_and_unload()
        merged.save_pretrained(str(output), safe_serialization=True)
        tokenizer.save_pretrained(str(output))

    # Preserve the base tokenizer_config.json (chat_template + special tokens).
    base_tok = base / "tokenizer_config.json"
    if base_tok.exists():
        shutil.copy(str(base_tok), str(output / "tokenizer_config.json"))
        print(f"Copied tokenizer_config.json from {base_tok}")

    print(f"\n✓ Merged model ready at: {output}")


if __name__ == "__main__":
    sys.exit(main())
