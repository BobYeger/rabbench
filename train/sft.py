#!/usr/bin/env python3
"""
RabBench SFT v2 — anti-forgetting protocol on Qwen3.5-9B-Base (Unsloth + TRL).

What changed from v1 (which lost 12.6pp on the factual probe):
  * 80% domain + 20% general replay (consume `train/prepare_data.py` output)
  * LR 1e-5 (was 2e-4) — domain adaptation, not from-scratch instruction tuning
  * 1 epoch cap with early stopping on (a) eval-loss plateau or (b) probe regression
  * Factual probe runs in-process every save_steps via FactualProbeCallback
  * Pilot mode: --pilot-steps 200 to validate the loop end-to-end before a full run

Usage:
  # 1) Build the mixed dataset once
  /home/aigroup/training-env/bin/python train/prepare_data.py

  # 2) Smoke test (200 steps, ~10 min on GB10)
  /home/aigroup/training-env/bin/python train/sft.py --pilot-steps 200

  # 3) Full run
  /home/aigroup/training-env/bin/python train/sft.py

Hardware notes (DGX Spark, GB10, sm_121, CUDA 13):
  * No FlashAttention — Unsloth selects eager attention automatically.
  * 128GB unified memory — use HF gradient_checkpointing (not Unsloth's), per v1 perf.

TRL compat:
  * 0.22.x in the unsloth-spark Docker: max_seq_length, tokenizer=, remove_unused_columns=True
  * 0.24.x in the host venv:           max_length,      processing_class=
  Both branches handled below.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Cheap import — pulls only the callback dataclasses, not torch/transformers core.
from transformers.trainer_callback import TrainerCallback


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "train" / "configs" / "v2_mixed.json"
BENCH_DIR = REPO_ROOT / "bench"
PROBE_FILE = REPO_ROOT / "questions" / "factual_probe.json"


def load_config(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def parse_args(defaults: dict) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--base-model", default=defaults["base_model"])
    p.add_argument("--train-file", default=defaults["mixed_train_file"])
    p.add_argument("--val-file", default=defaults["mixed_val_file"])
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

    p.add_argument("--probe-every-steps", type=int, default=defaults.get("probe_every_steps", 250))
    p.add_argument("--probe-baseline", type=float, default=defaults.get("probe_baseline_accuracy", 0.767))
    p.add_argument("--probe-threshold-pp", type=float, default=defaults.get("probe_regression_threshold_pp", 3.0))
    p.add_argument("--no-probe", action="store_true", help="Skip the in-training factual probe")
    p.add_argument("--no-early-stop", action="store_true", help="Disable early stopping (probe + eval-loss)")

    p.add_argument("--pilot-steps", type=int, default=None,
                   help="Cap training at this many steps and disable merge — for end-to-end validation")
    p.add_argument("--seed", type=int, default=defaults["seed"])
    p.add_argument("--no-merge", action="store_true")
    p.add_argument("--no-responses-only", action="store_true")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--run-name", default=None)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


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
                out.append({"messages": msgs})
    return out


def main() -> None:
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
    log("RabBench SFT v2 — mixed-data, low-LR, probe-gated")
    log(f"Base:        {args.base_model}")
    log(f"Train:       {args.train_file}")
    log(f"Val:         {args.val_file}")
    log(f"Output:      {args.output_dir}")
    log(f"LoRA:        r={args.lora_r} alpha={args.lora_alpha} dropout={args.lora_dropout}")
    log(f"Batch:       {args.batch_size} x grad_accum {args.grad_accum} = eff {args.batch_size * args.grad_accum}")
    log(f"LR / epochs: {args.lr} / {args.epochs} (warmup_ratio={args.warmup_ratio})")
    log(f"Probe:       every {args.probe_every_steps} steps, baseline {args.probe_baseline:.1%}, "
        f"threshold {args.probe_threshold_pp:.1f}pp")
    if args.pilot_steps:
        log(f"PILOT MODE:  capping at {args.pilot_steps} steps; merge disabled")
    log("=" * 70)

    import torch
    log(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")

    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template, train_on_responses_only
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from transformers import EarlyStoppingCallback

    import trl as _trl
    _trl_ver = tuple(int(x) for x in _trl.__version__.split(".")[:2])
    log(f"trl {_trl.__version__}  (compat path: {'>=0.24' if _trl_ver >= (0, 24) else '<0.24'})")

    log("Loading base model with Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        load_in_16bit=True,
        dtype=torch.bfloat16,
    )

    tokenizer = get_chat_template(tokenizer, chat_template=defaults.get("chat_template", "qwen-2.5"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log("Applying LoRA adapter...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=defaults["lora_target_modules"],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing=True,
        use_rslora=False,
        random_state=args.seed,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    log("Loading SFT data...")
    train_records = load_jsonl_messages(Path(args.train_file))
    log(f"  train: {len(train_records)} records")
    val_records: list[dict] = []
    if args.val_file and Path(args.val_file).exists():
        val_records = load_jsonl_messages(Path(args.val_file))
        log(f"  val:   {len(val_records)} records")

    def format_chat(example: dict) -> dict:
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False,
        )
        return {"text": text}

    train_ds = Dataset.from_list(train_records).map(format_chat, batched=False, remove_columns=["messages"])
    eval_ds = (
        Dataset.from_list(val_records).map(format_chat, batched=False, remove_columns=["messages"])
        if val_records else None
    )

    report_to = "wandb" if args.wandb else "none"
    run_name = args.run_name or f"rabbench-sft-v2-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    eval_strategy = "steps" if eval_ds is not None else "no"

    seq_kwarg = "max_length" if _trl_ver >= (0, 24) else "max_seq_length"
    seq_kwargs = {seq_kwarg: args.max_seq_length}

    sft_kwargs = dict(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
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
        load_best_model_at_end=eval_ds is not None and not args.no_early_stop,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        neftune_noise_alpha=defaults.get("neftune_noise_alpha", 0) or None,
        dataset_num_proc=1,
        dataloader_num_workers=0,
        seed=args.seed,
        report_to=report_to,
        run_name=run_name,
        remove_unused_columns=True,
        **seq_kwargs,
    )
    if args.pilot_steps:
        sft_kwargs["max_steps"] = args.pilot_steps

    training_args = SFTConfig(**sft_kwargs)

    trainer_kwargs: dict = {}
    if _trl_ver >= (0, 24):
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=training_args,
        **trainer_kwargs,
    )

    if not args.no_responses_only:
        trainer = train_on_responses_only(
            trainer,
            instruction_part=defaults.get("instruction_part", "<|im_start|>user\n"),
            response_part=defaults.get("response_part", "<|im_start|>assistant\n"),
        )
        log("Applied train_on_responses_only masking")

    if eval_ds is not None and not args.no_early_stop:
        patience = defaults.get("early_stop_eval_loss_patience", 2)
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=patience))
        log(f"Eval-loss early stopping enabled (patience={patience})")

    if not args.no_probe:
        sys.path.insert(0, str(BENCH_DIR))
        from probe import fuzzy_match, all_gold_variants, SHORT_SYSTEM_HE, build_user_prompt
        cb = FactualProbeCallback(
            model=model,
            tokenizer=tokenizer,
            probe_path=PROBE_FILE,
            every_steps=args.probe_every_steps,
            baseline_accuracy=args.probe_baseline,
            regression_pp=args.probe_threshold_pp,
            max_new_tokens=defaults.get("probe_max_new_tokens", 96),
            log_fn=log,
            output_dir=output_dir,
            stop_on_regression=not args.no_early_stop,
            fuzzy_match=fuzzy_match,
            all_gold_variants=all_gold_variants,
            system_prompt=SHORT_SYSTEM_HE,
            build_user_prompt=build_user_prompt,
        )
        trainer.add_callback(cb)
        log(f"FactualProbeCallback installed (every {args.probe_every_steps} steps, "
            f"baseline {args.probe_baseline:.1%}, stop@-{args.probe_threshold_pp:.1f}pp)")

    log("Starting training...")
    t0 = time.time()
    result = trainer.train(resume_from_checkpoint=args.resume or None)
    elapsed = time.time() - t0
    log(f"Training complete: {elapsed / 3600:.2f}h, final_loss={result.training_loss:.4f}")

    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    log(f"Adapter saved: {adapter_dir}")

    merged_dir = output_dir / "merged"
    if not args.no_merge and not args.pilot_steps:
        log("Merging LoRA -> bf16...")
        merged_dir.mkdir(parents=True, exist_ok=True)
        try:
            model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
        except Exception as e:
            log(f"Unsloth merge failed ({e}); falling back to peft merge_and_unload")
            merged = model.merge_and_unload()
            merged.save_pretrained(str(merged_dir))
            tokenizer.save_pretrained(str(merged_dir))
        import shutil
        base_tok = Path(args.base_model) / "tokenizer_config.json"
        if base_tok.exists():
            shutil.copy(str(base_tok), str(merged_dir / "tokenizer_config.json"))
        log(f"Merged: {merged_dir}")

    stats = {
        "phase": "sft_v2",
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
        "pilot_steps": args.pilot_steps,
        "train_on_responses_only": not args.no_responses_only,
        "probe_every_steps": args.probe_every_steps,
        "probe_baseline": args.probe_baseline,
        "probe_threshold_pp": args.probe_threshold_pp,
        "final_loss": float(result.training_loss),
        "elapsed_hours": elapsed / 3600,
        "adapter_dir": str(adapter_dir),
        "merged_dir": str(merged_dir) if (not args.no_merge and not args.pilot_steps) else None,
        "run_name": run_name,
        "completed": datetime.now(timezone.utc).isoformat(),
    }
    with open(output_dir / "training_stats.json", "w") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)
    log(f"Stats: {output_dir / 'training_stats.json'}")

    del trainer, model
    torch.cuda.empty_cache()
    log("Done.")


# ---------------------------------------------------------------------------
# In-training factual probe
# ---------------------------------------------------------------------------

class FactualProbeCallback(TrainerCallback):
    """
    Every `every_steps`, runs the 103-question factual probe in-process — no
    vLLM, no LLM judge, fuzzy match only (matches the deterministic path of
    bench/probe.py). Logs to probe_history.json. If accuracy drops more than
    `regression_pp` below `baseline_accuracy`, sets should_training_stop.
    """

    def __init__(self, *, model, tokenizer, probe_path: Path, every_steps: int,
                 baseline_accuracy: float, regression_pp: float,
                 max_new_tokens: int, log_fn, output_dir: Path,
                 stop_on_regression: bool,
                 fuzzy_match, all_gold_variants, system_prompt: str,
                 build_user_prompt) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.every_steps = every_steps
        self.baseline = baseline_accuracy
        self.threshold = regression_pp / 100.0
        self.max_new_tokens = max_new_tokens
        self.log = log_fn
        self.history_path = output_dir / "probe_history.json"
        self.stop_on_regression = stop_on_regression
        self.fuzzy_match = fuzzy_match
        self.all_gold_variants = all_gold_variants
        self.system_prompt = system_prompt
        self.build_user_prompt = build_user_prompt
        with open(probe_path) as f:
            self.questions = sorted(json.load(f)["questions"], key=lambda q: q["id"])
        self.history: list[dict] = []

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step <= 0 or state.global_step % self.every_steps != 0:
            return control
        return self._run_probe(state, control)

    def on_train_end(self, args, state, control, **kwargs):
        return self._run_probe(state, control, force=True)

    def _run_probe(self, state, control, force: bool = False):
        import torch
        try:
            from unsloth import FastLanguageModel
            FastLanguageModel.for_inference(self.model)
        except Exception:
            pass
        was_training = self.model.training
        self.model.eval()

        n_correct = 0
        n_total = 0
        t0 = time.time()
        device = next(self.model.parameters()).device

        with torch.no_grad():
            for q in self.questions:
                msgs = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": self.build_user_prompt(q)},
                ]
                prompt = self.tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True,
                )
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                         max_length=2048).to(device)
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                gen = out[0, inputs["input_ids"].shape[1]:]
                answer = self.tokenizer.decode(gen, skip_special_tokens=True).strip()
                matched, _reason = self.fuzzy_match(answer, self.all_gold_variants(q))
                n_total += 1
                if matched:
                    n_correct += 1

        accuracy = n_correct / max(1, n_total)
        elapsed = time.time() - t0
        delta_pp = (accuracy - self.baseline) * 100
        entry = {
            "step": state.global_step,
            "accuracy": round(accuracy, 4),
            "n_correct": n_correct,
            "n_total": n_total,
            "delta_pp_vs_baseline": round(delta_pp, 2),
            "elapsed_s": round(elapsed, 1),
        }
        self.history.append(entry)
        try:
            with open(self.history_path, "w") as f:
                json.dump({"baseline": self.baseline, "history": self.history}, f, indent=2)
        except Exception as e:
            self.log(f"  probe: failed to write history ({e})")

        tag = "OK" if delta_pp >= -self.threshold * 100 else "REGRESSION"
        self.log(
            f"  probe @ step {state.global_step}: {accuracy:.1%} "
            f"({n_correct}/{n_total})  Δ {delta_pp:+.1f}pp vs baseline  [{tag}]  "
            f"({elapsed:.0f}s)"
        )

        if self.stop_on_regression and delta_pp < -self.threshold * 100:
            self.log(f"  probe regression > {self.threshold * 100:.1f}pp — requesting training stop")
            control.should_training_stop = True

        if was_training:
            self.model.train()
            try:
                from unsloth import FastLanguageModel
                FastLanguageModel.for_training(self.model)
            except Exception:
                pass
        return control


if __name__ == "__main__":
    os.environ.setdefault("WANDB_MODE", "online" if os.environ.get("WANDB_API_KEY") else "disabled")
    main()
