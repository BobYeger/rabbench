# RabBench

Evaluation-first benchmark for Hebrew rabbinical LLMs.
**825 open-ended essay questions** from Israel's Chief Rabbinate certification exams (הסמכה לרבנות + דיינות). 15 subjects, 2 levels, 1,654 sub-questions. LLM-as-judge scoring against a 5-dimension rubric.

> No reference answers exist. This is the hard case: graders judge absolute quality against the rubric, not against a gold standard.

## Why this repo exists

V1-V8 of our Hebrew rabbinical LLM taught us one thing: **you can't iterate on training without a trustworthy eval**. Direct QA testing gave noisy, confounded signals. CPT eval was conflated with instruction-following. This repo fixes that: **build the benchmark first, prove it discriminates, then train against it**.

## Status

- Questions: ✅ 825 parsed, 15 subjects, Hebrew text
- Rubric: ✅ 5 dimensions (accuracy, sources, reasoning, completeness, language)
- Scripts: ✅ `generate.py`, `judge.py`, `compare.py`
- Baselines: ⏳ Sonnet 4.6 (pilot), Opus 4.6 (pilot), Qwen3.5-9B-Base (TBD)
- Judge stability: ⏳ to validate with pilot

## Layout

```
rabbench/
├── questions/
│   ├── benchmark_questions.json   # 825 questions
│   └── rubric.md                  # scoring rubric
├── bench/
│   ├── generate.py                # model → answers
│   ├── judge.py                   # Opus → scores
│   ├── compare.py                 # side-by-side report
│   ├── rabbench_io.py             # shared I/O + API key loader
│   └── configs/default.json
├── results/                       # gitignored
└── docs/PLAN.md
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

## Design principles

1. **Eval before training.** No training code in this repo.
2. **Absolute scoring.** No reference answers — the judge grades against the rubric.
3. **Transparent failures.** `red_flags` explicitly track fabricated citations, language drift, refusals.
4. **Reproducible runs.** Every output stamped with UTC time, model, provider, and full config.
5. **Deterministic ordering.** Questions sorted by `id` so pilot subsets are consistent.
