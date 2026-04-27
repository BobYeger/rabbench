# RabBench Training v2 — Anti-Forgetting Protocol

**Goal:** SFT that improves essay quality on the 825-question RabBench *without*
losing the factual accuracy the base model already has.

## Why v1 failed

| metric | Qwen3.5-9B-Base | v1 (3ep, lr=2e-4, domain-only) | Δ |
|---|---|---|---|
| factual probe (103 Qs) | 76.7% | 64.1% | **-12.6pp** |
| essay quality (judge)  | baseline | +0.85 | + |

Three things went wrong:

1. **Domain-only data.** Without general replay, every gradient step pushes the
   model further from the distribution it was pretrained on. Catastrophic
   forgetting is the textbook outcome (LIMA, FLAN-T5, OpenAssistant all show this).
2. **Far-too-high LR.** 2e-4 is from-scratch instruction-tuning territory.
   For domain adaptation on top of a model that already speaks Hebrew and knows
   halacha, 1e-5 is the right scale (per PaLM 2 / Mistral ablations).
3. **Eval loss plateaued at epoch 1.2; we ran 3 epochs.** Past the plateau,
   training was just memorizing noise + over-writing prior knowledge.

## v2 protocol — what changes

| dimension | v1 | v2 | reason |
|---|---|---|---|
| data | 100% domain | **80% domain + 20% replay** (UltraChat-200k) | preserves base distribution |
| LR | 2e-4 | **1e-5** | domain adaptation, not from-scratch |
| epochs | 3 | **≤ 1** with early stopping | eval loss plateaued at 1.2 ep in v1 |
| probe | only after | **every 250 steps**, hard stop on -3pp | catches forgetting in flight |
| neftune | α=5 | **off** | regularization not needed at this LR |
| warmup | 3% | **5%** | smoother low-LR warmup |
| val set | domain only | mixed (matches train) | eval loss reflects real objective |

## Files

```
train/
├── prepare_data.py            # build 80/20 mixed jsonl from existing domain SFT + UltraChat
├── sft.py                     # Unsloth + TRL trainer with FactualProbeCallback
├── eval_during_training.py    # run probe.py against a checkpoint (vLLM)
├── configs/v2_mixed.json      # all hyperparameters, paths, thresholds
└── README.md                  # this file
```

The factual probe (`bench/probe.py`, 103 Hebrew short-answer questions with gold
answers) is the contract: any v2 run that drops > 3pp below baseline aborts.

## Run it

```bash
# 0) Sanity-check sizes and replay split (no downloads, no training)
/home/aigroup/training-env/bin/python train/prepare_data.py --dry-run

# 1) Build the mixed dataset (~5 min, streams UltraChat-200k from HF Hub)
/home/aigroup/training-env/bin/python train/prepare_data.py

# 2) Pilot — 200 steps, ~10–15 min on GB10. Validates the loop end-to-end:
#    data load, LoRA init, masking, loss flow, probe callback, checkpoint save.
/home/aigroup/training-env/bin/python train/sft.py --pilot-steps 200

# 3) Inspect probe trajectory in the pilot:
cat /home/aigroup/training/rabbench-sft-v2/probe_history.json

# 4) Full run (1 epoch on ~24.9k mixed samples).
/home/aigroup/training-env/bin/python train/sft.py
```

The trainer saves a checkpoint every 250 steps. The factual probe runs
**in-process** at every save_step (uses `FastLanguageModel.for_inference`
from Unsloth, fuzzy match only — no LLM judge, no vLLM). If accuracy drops
> 3pp below the 76.7% baseline at any checkpoint, training aborts.

After training, validate the merged model end-to-end:

```bash
# Serve via vLLM (in a separate terminal/container)
docker run --rm -p 8000:8000 --gpus all \
    -v /home/aigroup/training/rabbench-sft-v2/merged:/model \
    vllm-openai:qwen3_5-hardened --model /model --enforce-eager \
    --served-model-name rabbench-sft-v2

# Run probe + compare to baseline
python3 train/eval_during_training.py \
    --model rabbench-sft-v2 \
    --base-url http://localhost:8000/v1 \
    --baseline results/probe_Qwen3.5-9B_latest.json \
    --threshold-pp 3.0
```

## TRL version compatibility

Two environments produce different TRL versions:

| env | TRL | seq kwarg | tokenizer kwarg |
|---|---|---|---|
| `unsloth-spark-v5:patched` Docker | 0.22.x | `max_seq_length=` | `tokenizer=` |
| `/home/aigroup/training-env` venv | 0.24.x | `max_length=`     | `processing_class=` |

`sft.py` detects `trl.__version__` and uses the right kwargs automatically.
Either env works; the Docker image is preferred because it ships FLA +
causal-conv1d ARM64 wheels patched for sm_121.

## Knobs you might want

- `--pilot-steps N` — cap at N steps, skip merge. Use this before any change.
- `--no-probe` — skip the in-training probe (faster, but flying blind).
- `--no-early-stop` — disable both probe-based and eval-loss-based stops.
- `--probe-every-steps 100` — more frequent probing for tighter monitoring.
- `--probe-threshold-pp 1.5` — stricter regression budget.
- Edit `configs/v2_mixed.json` for LR / replay ratio / LoRA rank changes.

## Decision log

- **Why UltraChat-200k for replay, not SlimOrca?** UltraChat ships native
  `messages` format (matches our domain data), is multi-turn (closer to real use),
  and has clean assistant turns. SlimOrca is single-turn instruction-following —
  fine but less representative of base-model strengths we want to preserve.
- **Why fuzzy match only in the in-training probe (no LLM judge)?** Speed and
  cost. The probe runs every 250 steps; an LLM judge would add ~$0.20 + 60s per
  call. The standalone `bench/probe.py` keeps the judge fallback for definitive
  post-hoc eval.
- **Why HF gradient checkpointing instead of Unsloth's?** v1 measurement: on
  GB10 / unified memory, Unsloth's "unsloth" mode was ~5x slower. Standard HF
  is the right default here.
