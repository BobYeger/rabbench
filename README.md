# RabBench

Evaluation-first benchmark for Hebrew rabbinical LLMs.
**825 open-ended essay questions** from Israel's Chief Rabbinate certification exams (הסמכה לרבנות + דיינות). 15 subjects, 2 levels, 1,654 sub-questions. LLM-as-judge scoring against a 5-dimension rubric.

> No reference answers exist. This is the hard case: graders judge absolute quality against the rubric, not against a gold standard.

## Why this repo exists

V1-V8 of our Hebrew rabbinical LLM taught us one thing: **you can't iterate on training without a trustworthy eval**. Direct QA testing gave noisy, confounded signals. CPT eval was conflated with instruction-following. This repo fixes that: **build the benchmark first, prove it discriminates, then train against it**.

## Status

- Questions: ✅ 825 essay + 103 factual probe (with gold answers)
- Rubric: ✅ 5 dimensions (accuracy, sources, reasoning, completeness, language)
- Bench scripts: ✅ `generate.py`, `judge.py`, `compare.py`, `probe.py`
- Baselines: ✅ Sonnet 4.6, Opus 4.6, Haiku 4.5, GPT-4o-mini, Qwen3.5-9B-Base
- Training v1 (lr=2e-4, domain-only, 3 ep): **❌ -12.6pp on factual probe** (catastrophic forgetting)
- Training v2 (mixed data, lr=1e-5, 1 ep, probe-gated): 🚧 implemented, not yet run

## What we learned from v1

V1 SFT improved essay quality by +0.85 on the rubric but dropped factual probe
accuracy from 76.7% → 64.1%. Eval loss plateaued at epoch 1.2; epochs 2–3 were
pure forgetting. The full post-mortem and the v2 protocol live in
[`train/README.md`](train/README.md).

## Layout

```
rabbench/
├── questions/
│   ├── benchmark_questions.json   # 825 essay questions
│   ├── factual_probe.json         # 103 short-answer Qs with gold answers
│   └── rubric.md                  # scoring rubric
├── bench/
│   ├── generate.py                # model → essay answers
│   ├── judge.py                   # Opus → rubric scores
│   ├── compare.py                 # side-by-side report
│   ├── probe.py                   # factual probe runner (fuzzy + LLM judge)
│   ├── rabbench_io.py             # shared I/O + API key loader
│   └── configs/default.json
├── train/
│   ├── prepare_data.py            # build 80/20 domain+replay mix
│   ├── sft.py                     # Unsloth + TRL SFT, in-process probe callback
│   ├── eval_during_training.py    # check a checkpoint via vLLM probe
│   ├── configs/v2_mixed.json
│   └── README.md                  # v2 protocol & v1 post-mortem
├── results/                       # gitignored
```

## Quick start

API key lives in `~/.openclaw/workspace/memory/api-keys.md` under `## Anthropic API`. `generate.py` and `judge.py` read it automatically.

```bash
cd bench

# 1. Pilot: generate 20 answers with Sonnet
python3 generate.py --model claude-sonnet-4-6 --pilot 20

# 2. Judge those 20 with Opus
python3 judge.py --input ../results/gen_claude-sonnet-4-6_latest.json

# 3. Baseline Opus too (reference ceiling)
python3 generate.py --model claude-opus-4-6 --pilot 20
python3 judge.py --input ../results/gen_claude-opus-4-6_latest.json

# 4. Compare
python3 compare.py ../results/judge_*_latest.json
```

For local Qwen3.5 base via vLLM:
```bash
python3 generate.py --model Qwen3.5-9B-Base \
    --provider openai --base-url http://localhost:8000/v1 \
    --base-model --pilot 20
```

## Rubric (summary)

| Dimension | Range | Weight |
|---|---|---|
| דיוק הלכתי (accuracy) | 1-5 | 1.0 |
| ציון מקורות (sources) | 1-5 | 1.0 |
| עומק הנימוק (reasoning) | 1-5 | 1.0 |
| שלמות (completeness) | 1-5 | 1.0 |
| איכות לשון (language) | 1-5 | 0.5 |

Max weighted total: **22.5**. See [`questions/rubric.md`](questions/rubric.md) for full criteria.

## Cost estimates (Anthropic direct)

Based on Anthropic pricing (Sonnet 4.6 $3/MTok in, $15/MTok out; Opus 4.6 $15/MTok in, $75/MTok out):

| Run | Target | Est. cost |
|---|---|---|
| Pilot (20 Qs) | Sonnet gen + Opus judge | ~$2 |
| Full (825 Qs) | Sonnet gen | ~$30 |
| Full (825 Qs) | Opus judge | ~$65 |
| Full (825 Qs) | Opus gen (reference) | ~$150 |

Local models: $0 compute, use our own GPU.

## Training

SFT on top of `Qwen3.5-9B-Base` with Unsloth + TRL on DGX Spark (GB10, sm_121,
no FlashAttention). The v2 protocol fixes v1's forgetting collapse: 80/20
domain+replay mix, LR 1e-5, 1-epoch cap, factual-probe-gated early stopping.

```bash
# 1) Build 80/20 mixed jsonl (domain SFT + UltraChat-200k replay)
/home/aigroup/training-env/bin/python train/prepare_data.py

# 2) Pilot — 200 steps end-to-end check with the in-process factual probe
/home/aigroup/training-env/bin/python train/sft.py --pilot-steps 200

# 3) Full run (1 epoch, ~24.9k samples)
/home/aigroup/training-env/bin/python train/sft.py
```

Hyperparameters live in [`train/configs/v2_mixed.json`](train/configs/v2_mixed.json).
Full protocol, decision log, and post-hoc validation steps are in
[`train/README.md`](train/README.md).

## Design principles

1. **Eval-first.** Benchmark design came before training code — all `train/` outputs flow back through `bench/`.
2. **Absolute scoring.** No reference answers — the judge grades against the rubric.
3. **Transparent failures.** `red_flags` explicitly track fabricated citations, language drift, refusals.
4. **Reproducible runs.** Every output stamped with UTC time, model, provider, and full config.
5. **Deterministic ordering.** Questions sorted by `id` so pilot subsets are consistent.
