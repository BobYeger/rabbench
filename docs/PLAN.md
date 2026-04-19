# RabBench — Plan

## Why we're starting over

V1–V8 of the Hebrew rabbinical LLM produced training runs we couldn't trust because the evaluation was:
- Confounded with instruction-following (CPT eval returning gibberish ≠ no knowledge)
- Graded against a noisy, inconsistent QA set we built ourselves
- Using weak metrics (raw accuracy, BLEU) that miss the thing that matters: **can this answer a rabbinic exam question?**

The fix is eval-first: **build the benchmark, prove it discriminates, then iterate training against it.**

## Phase 0 — Setup  ✅ *done*

- Fresh repo: `rabbench/`
- Questions copied in (825, 15 subjects, rabbinical + dayan levels)
- Rubric written
- Scripts drafted (`generate.py`, `judge.py`, `compare.py`)
- API key: direct Anthropic (not OAuth, not OpenRouter)

## Phase 1 — Pilot  ⏳ *next*

Goal: confirm the benchmark discriminates and the judge is consistent.

1. Generate 20 pilot answers with **Sonnet 4.6**
2. Judge with **Opus 4.6**
3. Inspect: are scores spread? Is the judge citing specific rubric clauses? Are rationales in Hebrew?
4. Generate 20 with **Opus 4.6** (reference ceiling)
5. Compare Sonnet vs Opus
   - Expectation: Opus ≳ Sonnet on accuracy + sources + reasoning
   - If the gap is <1 point on avg weighted: rubric isn't discriminating well
6. Re-judge the same 20 Sonnet answers a second time to measure judge stability
   - Expectation: avg dim scores within ±0.3 between runs

**Success criteria to proceed to Phase 2:**
- Judge parses JSON reliably (>95% of calls parse)
- Opus scores at least 1.0 point higher weighted than Sonnet on average
- Judge stability within ±0.3 on re-run
- Hebrew rationales that reference specific rubric dimensions

**If any fail:** tighten the rubric prompt, iterate before scaling to 825.

## Phase 2 — Baselines  (after pilot green)

1. Full 825 with **Sonnet 4.6** (~$30)
2. Full 825 with **Opus 4.6** (~$150 — decide if worth it, maybe skip)
3. Full 825 with **GPT-4o** (parity check for judge bias)
4. Full 825 with **Qwen3.5-9B-Base** (our base, chat template off)
5. Full 825 with **Qwen3.5-9B-Instruct** (chat template on)

Output: a leaderboard across external SOTA, our base, and our base-instruct.

## Phase 3 — Judge validation

Risks to address:
- **Judge bias:** does Opus prefer answers stylistically similar to itself? Test by running a second judge (GPT-4o or a separate Opus session with different system prompt) on a subset.
- **Self-preference:** never let a model judge its own answers. Opus-as-generator must be scored by a non-Opus judge.
- **Sefer-name dropping:** models may cite plausible-sounding non-existent sources. `red_flags.fabricated_citation` tracks this; spot-check by hand.

## Phase 4 — Training targets

Only after Phases 1–3 show a stable, meaningful gap between base Qwen and SOTA:

- Define what we're optimizing: weighted total, specific dimensions, or per-subject gaps
- Re-run CPT/SFT decisions against THIS benchmark (not perplexity, not MCQ)
- Each training iteration: `generate.py` → `judge.py` → `compare.py` vs previous
- No training iteration lands without measurable benchmark delta

## Open decisions

- [ ] Do we run full Opus baseline ($150) or trust pilot?
- [ ] Judge bias mitigation: add GPT-4o as secondary judge on a 100-question sample?
- [ ] Human spot-check: should we hand-review 20 judgments to validate Opus's scoring?
- [ ] Should the `benchmark_questions.json` have train/dev/test splits so we don't leak during future training?

## Non-goals for this repo

- No training code
- No model weights
- No data factory / scraping
- No serving / inference stack

Those live elsewhere. This repo only answers: **how good is model X at the Chief Rabbinate exam?**
