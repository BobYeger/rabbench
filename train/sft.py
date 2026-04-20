#!/usr/bin/env python3
"""
SFT training for the Hebrew rabbinical LLM — Unsloth + TRL on Qwen3.5-9B.

Produces a LoRA adapter and a merged bf16 model ready for vLLM serving.

Usage:
  # Defaults from train/configs/sft_default.json
  /home/aigroup/training-env/bin/python train/sft.py

  # Override via CLI
  /home/aigroup/training-env/bin/python train/sft.py \\
      --train-file /path/to/sft_train.jsonl \\
      --output-dir /home/aigroup/training/rabbench-sft-v1 \\
      --epochs 3 --batch-size 2 --grad-accum 8 --lr 2e-4 \\
      --wandb

Hardware notes (DGX Spark):
  - GB10 (sm_121, CUDA 13) — no FlashAttention. Uses attn_implementation=eager.
  - 128GB unified memory — avoid "unsloth" gradient checkpointing (5x slower here);
    use standard HF gradient checkpointing instead.

Data format (each JSONL line):
  {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}],
   "source": "...", "tier": "gold"|"bronze"}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "train" / "configs" / "sft_default.json"


def load_config(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def parse_args(defaults: dict) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="Config JSON (provides defaults)")
    p.add_argument("--base-model", default=defaults["base_model"])
    p.add_argument("--train-file", default=defaults["train_file"])
    p.add_argument("--val-file", default=defaults["val_file"])
    p.add_argument("--output-dir", default=defaults["output_dir"])

    p.add_argument("--max-seq-length", type=int, default=defaults["max_seq_length"])
    p.add_argument("--batch-size", type=int, default=defaults["per_device_batch"])
    p.add_argument("--grad-accum", type=int, default=defaults["grad_accum"])
    p.add_argument("--lr", type=float, default=defaults["learning_rate"])
    p.add_argument("--epochs", type=float, default=defaults["num_epochs"])
    p.add_argument("--warmup-ratio", type=float, default=defaults["warmup_ratio"])

    p.add_argument("--lora-r", type=int, default=defaults["lora_r"])
    p.add_argument("--lora-alpha", type=int, default=defaults["lora_alpha"])
    p.add_argument("--lora-dropout", type=float, default=defaults["lora_dropout"])

    p.add_argument("--logging-steps", type=int, default=defaults["logging_steps"])
    p.add_argument("--eval-steps", type=int, default=defaults["eval_steps"])
    p.add_argument("--save-steps", type=int, default=defaults["save_steps"])
    p.add_argument("--save-total-limit", type=int, default=defaults["save_total_limit"])

    p.add_argument("--seed", type=int, default=defaults["seed"])
    p.add_argument("--no-merge", action="store_true", help="Skip merge step after training")
    p.add_argument("--no-responses-only", action="store_true", help="Disable train_on_responses_only masking")

    p.add_argument("--wandb", action="store_true", help="Log to Weights & Biases (set WANDB_API_KEY)")
    p.add_argument("--run-name", default=None, help="W&B / HF run name")
    p.add_argument("--resume", action="store_true", help="Resume from last checkpoint in output_dir")

    return p.parse_args()


def load_jsonl_messages(path: Path) -> list[dict]:
    """Parse JSONL and keep only records with a non-empty messages list."""
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
                out.append({"messages": msgs})
    return out


def main() -> None:
    # Parse args
    ap_tmp = argparse.ArgumentParser(add_help=False)
    ap_tmp.add_argument("--config", default=str(DEFAULT_CONFIG))
    cfg_only, _ = ap_tmp.parse_known_args()
    defaults = load_config(Path(cfg_only.config))
    args = parse_args(defaults)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / "sft_training.log"

    def log(msg: str) -> None:
        line = f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_file, "a") as fh:
            fh.write(line + "\n")

    log("=" * 70)
    log("RabBench SFT — Qwen3.5-9B + Unsloth + TRL")
    log(f"Base model:   {args.base_model}")
    log(f"Train file:   {args.train_file}")
    log(f"Val file:     {args.val_file}")
    log(f"Output:       {args.output_dir}")
    log(f"LoRA:         r={args.lora_r} alpha={args.lora_alpha} dropout={args.lora_dropout}")
    log(f"Batch:        {args.batch_size} × grad_accum {args.grad_accum} = "
        f"effective {args.batch_size * args.grad_accum}")
    log(f"LR/epochs:    {args.lr} / {args.epochs} (warmup_ratio={args.warmup_ratio})")
    log(f"Max seq len:  {args.max_seq_length}")
    log("=" * 70)

    # Imports inside main so --help is cheap
    import torch
    log(f"torch: {torch.__version__}  CUDA avail: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")

    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template, train_on_responses_only
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    # ---- Load base model ----------------------------------------------------
    log("Loading base model with Unsloth...")
    # Unsloth auto-selects the attention backend per arch. On GB10 (sm_121) this
    # falls back to eager since FlashAttention isn't supported.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        load_in_16bit=True,
        dtype=torch.bfloat16,
    )

    # Qwen3.5 ships its own chat template; Unsloth's "qwen-2.5" template uses the
    # same <|im_start|>/<|im_end|> markers, which is what we mask on later.
    tokenizer = get_chat_template(tokenizer, chat_template=defaults.get("chat_template", "qwen-2.5"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Apply LoRA ---------------------------------------------------------
    log("Applying LoRA adapter...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=defaults["lora_target_modules"],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing=True,  # standard HF: faster on unified memory than "unsloth"
        use_rslora=False,
        random_state=args.seed,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # ---- Load data ----------------------------------------------------------
    log("Loading SFT data...")
    train_records = load_jsonl_messages(Path(args.train_file))
    log(f"  train: {len(train_records)} records")
    val_records: list[dict] = []
    if args.val_file and Path(args.val_file).exists():
        val_records = load_jsonl_messages(Path(args.val_file))
        log(f"  val:   {len(val_records)} records")

    def format_chat(example: dict) -> dict:
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    train_ds = Dataset.from_list(train_records).map(format_chat, batched=False, remove_columns=["messages"])
    eval_ds = (
        Dataset.from_list(val_records).map(format_chat, batched=False, remove_columns=["messages"])
        if val_records else None
    )

    # ---- Training args ------------------------------------------------------
    report_to = "wandb" if args.wandb else "none"
    run_name = args.run_name or f"rabbench-sft-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    eval_strategy = "steps" if eval_ds is not None else "no"

    training_args = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_length=args.max_seq_length,  # TRL 0.24+ renamed max_seq_length → max_length
        dataset_text_field="text",
        packing=False,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=defaults.get("lr_scheduler", "cosine"),
        weight_decay=defaults.get("weight_decay", 0.0),
        max_grad_norm=defaults.get("max_grad_norm", 1.0),
        bf16=True,
        fp16=False,
        logging_steps=args.logging_steps,
        eval_strategy=eval_strategy,
        eval_steps=args.eval_steps if eval_ds is not None else None,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        neftune_noise_alpha=defaults.get("neftune_noise_alpha", 5),
        dataset_num_proc=1,
        dataloader_num_workers=0,
        seed=args.seed,
        report_to=report_to,
        run_name=run_name,
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,  # TRL 0.24 renamed tokenizer → processing_class
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=training_args,
    )

    # Mask loss on prompt tokens — only learn from the assistant turn.
    # This was a +3pp jump in prior runs; skip only with --no-responses-only for debugging.
    if not args.no_responses_only:
        trainer = train_on_responses_only(
            trainer,
            instruction_part=defaults.get("instruction_part", "<|im_start|>user\n"),
            response_part=defaults.get("response_part", "<|im_start|>assistant\n"),
        )
        log("Applied train_on_responses_only masking")

    # ---- Train --------------------------------------------------------------
    log("Starting training...")
    t0 = time.time()
    result = trainer.train(resume_from_checkpoint=args.resume or None)
    elapsed = time.time() - t0
    log(f"Training complete: {elapsed / 3600:.2f}h, final_loss={result.training_loss:.4f}")

    # ---- Save adapter -------------------------------------------------------
    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    log(f"Adapter saved: {adapter_dir}")

    # ---- Save merged bf16 ---------------------------------------------------
    merged_dir = output_dir / "merged"
    if not args.no_merge:
        log("Merging LoRA into base model (bf16)...")
        merged_dir.mkdir(parents=True, exist_ok=True)
        try:
            model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
        except Exception as e:
            log(f"Unsloth merge failed ({e}); falling back to peft merge_and_unload")
            merged = model.merge_and_unload()
            merged.save_pretrained(str(merged_dir))
            tokenizer.save_pretrained(str(merged_dir))

        # Copy the base model's tokenizer_config.json so the chat_template survives.
        import shutil
        base_tok = Path(args.base_model) / "tokenizer_config.json"
        if base_tok.exists():
            shutil.copy(str(base_tok), str(merged_dir / "tokenizer_config.json"))
        log(f"Merged model: {merged_dir}")

    # ---- Stats --------------------------------------------------------------
    stats = {
        "phase": "sft",
        "base_model": args.base_model,
        "train_file": args.train_file,
        "val_file": args.val_file,
        "n_train": len(train_records),
        "n_val": len(val_records),
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_target_modules": defaults["lora_target_modules"],
        "max_seq_length": args.max_seq_length,
        "per_device_batch": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch": args.batch_size * args.grad_accum,
        "lr": args.lr,
        "epochs": args.epochs,
        "warmup_ratio": args.warmup_ratio,
        "train_on_responses_only": not args.no_responses_only,
        "final_loss": float(result.training_loss),
        "elapsed_hours": elapsed / 3600,
        "adapter_dir": str(adapter_dir),
        "merged_dir": str(merged_dir) if not args.no_merge else None,
        "run_name": run_name,
        "completed": datetime.now(timezone.utc).isoformat(),
    }
    with open(output_dir / "training_stats.json", "w") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)
    log(f"Stats: {output_dir / 'training_stats.json'}")

    # ---- Free GPU -----------------------------------------------------------
    del trainer, model
    torch.cuda.empty_cache()
    log("Done.")


if __name__ == "__main__":
    # Set a sane default so users without the W&B env var don't get prompted.
    os.environ.setdefault("WANDB_MODE", "online" if os.environ.get("WANDB_API_KEY") else "disabled")
    main()
