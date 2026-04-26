#!/usr/bin/env python3
"""
Run the RabBench factual probe — short-answer halachic questions with gold answers.

Tests pure factual knowledge (no style padding). Each question has one
unambiguous answer; we score with fuzzy matching first, then fall back to
an LLM judge for ambiguous cases.

Examples:
  python3 probe.py --model claude-sonnet-4-6
  python3 probe.py --model claude-sonnet-4-6 --pilot 20
  python3 probe.py --model gpt-4o --provider openai
  python3 probe.py --model my-qwen --provider openai \\
      --base-url http://localhost:8000/v1 --pilot 20
"""

import argparse
import asyncio
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import httpx

from rabbench_io import (
    BENCH_DIR,
    load_config,
    load_api_key,
    result_path,
    latest_symlink,
    utc_stamp,
)


PROBE_FILE = (BENCH_DIR.parent / "questions" / "factual_probe.json").resolve()


# ---------------------------------------------------------------------------
# Probe-specific prompts
# ---------------------------------------------------------------------------

SHORT_SYSTEM_HE = (
    "אתה מומחה הלכה. ענה תשובה קצרה ועניינית. "
    "תן רק את העובדה המבוקשת — מספר, שם, או משפט אחד קצר. "
    "אין צורך במקורות, נימוקים או הסברים."
)


def build_user_prompt(q: dict) -> str:
    return f"שאלה: {q['question_he']}\n\nתשובה קצרה:"


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

class AnthropicBackend:
    def __init__(self, model: str, api_key: str, cfg: dict):
        self.model = model
        self.api_key = api_key
        self.url = cfg["api"]["anthropic"]["base_url"]
        self.max_tokens = 256
        self.temperature = 0.0

    async def generate(self, client: httpx.AsyncClient, q: dict) -> dict:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "system": SHORT_SYSTEM_HE,
            "messages": [{"role": "user", "content": build_user_prompt(q)}],
        }
        t0 = time.time()
        r = await client.post(self.url, headers=headers, json=payload, timeout=120.0)
        r.raise_for_status()
        data = r.json()
        text = "".join(blk.get("text", "") for blk in data.get("content", []) if blk.get("type") == "text")
        usage = data.get("usage", {})
        return {
            "answer": text.strip(),
            "latency_s": round(time.time() - t0, 2),
            "usage": {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
            },
        }


class OpenAICompatibleBackend:
    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.max_tokens = 256
        self.temperature = 0.0

    async def generate(self, client: httpx.AsyncClient, q: dict) -> dict:
        headers = {
            "authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SHORT_SYSTEM_HE},
                {"role": "user", "content": build_user_prompt(q)},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        t0 = time.time()
        r = await client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload, timeout=300.0)
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"].get("content", "") or ""
        usage = data.get("usage", {}) or {}
        return {
            "answer": text.strip(),
            "latency_s": round(time.time() - t0, 2),
            "usage": {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
            },
        }


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

# Hebrew digit-letters → numeric value (gematria, partial — sufficient for small numbers)
GEMATRIA = {
    "א": 1, "ב": 2, "ג": 3, "ד": 4, "ה": 5, "ו": 6, "ז": 7, "ח": 8, "ט": 9,
    "י": 10, "כ": 20, "ך": 20, "ל": 30, "מ": 40, "ם": 40, "נ": 50, "ן": 50,
    "ס": 60, "ע": 70, "פ": 80, "ף": 80, "צ": 90, "ץ": 90,
    "ק": 100, "ר": 200, "ש": 300, "ת": 400,
}

HEBREW_NUMBER_WORDS = {
    "אחד": 1, "אחת": 1, "שניים": 2, "שתים": 2, "שתיים": 2, "שני": 2, "שתי": 2,
    "שלוש": 3, "שלושה": 3, "שלש": 3, "שלשה": 3,
    "ארבע": 4, "ארבעה": 4,
    "חמש": 5, "חמשה": 5, "חמישה": 5,
    "שש": 6, "ששה": 6, "שישה": 6, "שישית": 6, "שישי": 6,
    "שבע": 7, "שבעה": 7,
    "שמונה": 8, "שמונת": 8,
    "תשע": 9, "תשעה": 9,
    "עשר": 10, "עשרה": 10,
    "אחד-עשר": 11, "אחד עשר": 11, "אחת-עשרה": 11, "אחת עשרה": 11,
    "שתים-עשרה": 12, "שניים-עשר": 12, "שתים עשרה": 12, "שניים עשר": 12,
    "שלוש-עשרה": 13, "שלושה-עשר": 13, "שלוש עשרה": 13, "שלושה עשר": 13,
    "ארבע-עשרה": 14, "ארבעה-עשר": 14, "ארבע עשרה": 14, "ארבעה עשר": 14,
    "תשע-עשרה": 19, "תשעה-עשר": 19, "תשע עשרה": 19, "תשעה עשר": 19,
    "עשרים": 20, "שלושים": 30, "ארבעים": 40, "חמישים": 50,
    "שישים": 60, "שבעים": 70, "שמונים": 80, "תשעים": 90,
    "מאה": 100, "מאתיים": 200, "מאתים": 200,
    "אלף": 1000, "אלפיים": 2000, "אלפים": 2000,
}

ENGLISH_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000,
}


def _strip_diacritics(s: str) -> str:
    """Strip Hebrew nikud and other combining marks."""
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def normalize(s: str) -> str:
    s = _strip_diacritics(s)
    s = s.lower()
    s = re.sub(r"[\"'״׳`’]", "", s)
    s = re.sub(r"[^\w\s֐-׿]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def gematria_value(token: str) -> int | None:
    """Heuristic gematria for tokens like ל\"ט, י\"ט, ז'. Returns None if not a gematria token."""
    raw = re.sub(r"['\"׳״]", "", token)
    if not raw or not all("א" <= c <= "ת" for c in raw):
        return None
    if len(raw) > 4:  # words like שלושים are not gematria
        return None
    total = sum(GEMATRIA.get(c, 0) for c in raw)
    return total if total > 0 else None


def extract_numbers(text: str) -> set[int]:
    """Pull numeric values out of free text — Arabic digits, English/Hebrew words, gematria."""
    nums: set[int] = set()
    norm = normalize(text)

    for m in re.findall(r"\d+", norm):
        try:
            nums.add(int(m))
        except ValueError:
            pass

    tokens = norm.split()
    for t in tokens:
        if t in HEBREW_NUMBER_WORDS:
            nums.add(HEBREW_NUMBER_WORDS[t])
        if t in ENGLISH_NUMBER_WORDS:
            nums.add(ENGLISH_NUMBER_WORDS[t])

    # Multi-token Hebrew compounds: "תשע עשרה", "שלושים ותשע"
    for i in range(len(tokens) - 1):
        bigram = f"{tokens[i]} {tokens[i+1]}"
        if bigram in HEBREW_NUMBER_WORDS:
            nums.add(HEBREW_NUMBER_WORDS[bigram])
        # patterns like "שלושים ותשע" → 39
        if tokens[i] in HEBREW_NUMBER_WORDS and tokens[i+1].startswith("ו"):
            tail = tokens[i+1][1:]
            if tail in HEBREW_NUMBER_WORDS:
                nums.add(HEBREW_NUMBER_WORDS[tokens[i]] + HEBREW_NUMBER_WORDS[tail])

    # Gematria tokens (ל\"ט, י\"ט, etc.) — use original (pre-normalize) tokens too
    for raw_tok in re.findall(r"[֐-׿]+[\"'״׳]?[֐-׿]*", text):
        v = gematria_value(raw_tok)
        if v is not None:
            nums.add(v)

    return nums


def fuzzy_match(answer: str, gold_variants: list[str]) -> tuple[bool, str]:
    """
    Returns (matched, reason). Strategy:
      1. Numeric overlap: if any gold variant has numbers and answer covers all of them, match.
      2. Substring (normalized) of any gold variant appearing in the answer (or vice versa for short golds).
    """
    if not answer.strip():
        return False, "empty answer"

    a_norm = normalize(answer)
    a_nums = extract_numbers(answer)

    for gold in gold_variants:
        g_norm = normalize(gold)
        if not g_norm:
            continue
        g_nums = extract_numbers(gold)

        # Numeric path — useful when gold is "39" and model says "שלושים ותשע"
        if g_nums and g_nums.issubset(a_nums):
            return True, f"numeric match {sorted(g_nums)} via gold='{gold}'"

        # Substring path — short golds ("חליצה", "Rambam") that should appear verbatim
        if len(g_norm) <= 60 and g_norm in a_norm:
            return True, f"substring match via gold='{gold}'"

        # Token-overlap for slightly longer golds: ≥80% of gold tokens present in answer
        g_tokens = [t for t in g_norm.split() if len(t) >= 2]
        if g_tokens:
            hits = sum(1 for t in g_tokens if t in a_norm)
            if hits / len(g_tokens) >= 0.8 and len(g_tokens) >= 2:
                return True, f"token-overlap {hits}/{len(g_tokens)} via gold='{gold}'"

    return False, "no fuzzy match"


# ---------------------------------------------------------------------------
# LLM judge for ambiguous cases
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = (
    "You are a strict but fair judge of factual answers to halachic/rabbinic short-answer questions. "
    "You read Hebrew fluently. You return JSON only."
)

JUDGE_TEMPLATE = """Determine whether the candidate's answer is factually correct.

# QUESTION (Hebrew)
{question_he}

# QUESTION (English gloss)
{question_en}

# GOLD ANSWER
{gold_en}
Hebrew form: {gold_he}
Also acceptable phrasings: {accept_also}

# CANDIDATE ANSWER
{answer}

# RULES
- The candidate may answer in Hebrew or English; either is fine.
- Accept any phrasing that conveys the same fact (e.g., "שלושים ותשע" = "39" = "thirty-nine" = "ל\\"ט").
- Reject if the candidate gives a clearly different number, a different name, or refuses to answer.
- Extra elaboration is OK as long as the core fact is correct and unambiguous.
- If the answer states the right fact alongside contradictory facts, mark it incorrect.

# OUTPUT
Return a single JSON object (no markdown):
{{"correct": true|false, "extracted_answer": "<the part of the candidate's answer that's the actual answer>", "explanation": "<one short sentence>"}}"""


JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_judge_json(text: str) -> dict | None:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = JSON_OBJ_RE.search(text)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


async def judge_with_llm(client, cfg, judge_model, judge_key, q, answer) -> dict:
    accept = q.get("accept_also") or []
    payload = {
        "model": judge_model,
        "max_tokens": 400,
        "temperature": 0.0,
        "system": JUDGE_SYSTEM,
        "messages": [{
            "role": "user",
            "content": JUDGE_TEMPLATE.format(
                question_he=q["question_he"],
                question_en=q.get("question_en", ""),
                gold_en=q["gold_answer"],
                gold_he=q.get("gold_answer_he", ""),
                accept_also=", ".join(accept) if accept else "(none)",
                answer=answer,
            ),
        }],
    }
    headers = {
        "x-api-key": judge_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    url = cfg["api"]["anthropic"]["base_url"]

    last_err = None
    for attempt in range(3):
        try:
            r = await client.post(url, headers=headers, json=payload, timeout=120.0)
            r.raise_for_status()
            data = r.json()
            text = "".join(blk.get("text", "") for blk in data.get("content", []) if blk.get("type") == "text")
            parsed = parse_judge_json(text)
            if parsed and "correct" in parsed:
                return {
                    "correct": bool(parsed.get("correct")),
                    "extracted_answer": parsed.get("extracted_answer", ""),
                    "explanation": parsed.get("explanation", ""),
                    "judge_used": True,
                }
            last_err = f"unparseable judge output: {text[:200]}"
        except httpx.HTTPStatusError as e:
            last_err = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
        await asyncio.sleep(2 ** attempt)

    return {"correct": False, "extracted_answer": "", "explanation": f"judge failed: {last_err}", "judge_used": True, "judge_error": last_err}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def all_gold_variants(q: dict) -> list[str]:
    variants = [q["gold_answer"]]
    if q.get("gold_answer_he"):
        variants.append(q["gold_answer_he"])
    variants.extend(q.get("accept_also") or [])
    return variants


async def score_one(backend, judge_client, cfg, judge_model, judge_key, gen_client, q) -> dict:
    # Generate
    last_err = None
    gen_result = None
    for attempt in range(3):
        try:
            gen_result = await backend.generate(gen_client, q)
            break
        except httpx.HTTPStatusError as e:
            last_err = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
        await asyncio.sleep(2 ** attempt)

    if gen_result is None:
        return {
            "id": q["id"],
            "subject": q["subject"],
            "level": q["level"],
            "ok": False,
            "error": last_err,
            "correct": False,
        }

    answer = gen_result["answer"]

    # Fuzzy match first
    matched, reason = fuzzy_match(answer, all_gold_variants(q))

    judge_info: dict | None = None
    if matched:
        correct = True
    else:
        judge_info = await judge_with_llm(judge_client, cfg, judge_model, judge_key, q, answer)
        correct = judge_info["correct"]

    return {
        "id": q["id"],
        "subject": q["subject"],
        "level": q["level"],
        "question_he": q["question_he"],
        "gold_answer": q["gold_answer"],
        "answer": answer,
        "correct": correct,
        "match_path": "fuzzy" if matched else "judge",
        "match_reason": reason,
        "judge": judge_info,
        "ok": True,
        "latency_s": gen_result["latency_s"],
        "usage": gen_result["usage"],
    }


async def main_async(args, cfg):
    with open(PROBE_FILE) as f:
        probe = json.load(f)
    questions = sorted(probe["questions"], key=lambda q: q["id"])
    if args.subjects:
        wanted = set(args.subjects.split(","))
        questions = [q for q in questions if q["subject"] in wanted]
    if args.pilot:
        questions = questions[: args.pilot]

    print(f"→ {len(questions)} probe questions", file=sys.stderr)

    # Build generation backend
    if args.provider == "anthropic":
        gen_key = load_api_key("anthropic", cfg)
        backend = AnthropicBackend(args.model, gen_key, cfg)
    elif args.provider == "openai":
        gen_key = (args.api_key or "EMPTY") if args.base_url else load_api_key("openai", cfg)
        base_url = args.base_url or cfg["api"]["openai"]["base_url"]
        backend = OpenAICompatibleBackend(args.model, gen_key, base_url)
    else:
        raise ValueError(f"Unknown provider: {args.provider}")

    # Judge always uses Anthropic
    judge_model = args.judge_model or cfg["judge"]["model"]
    judge_key = load_api_key("anthropic", cfg)

    stamp = utc_stamp()
    out_path = result_path(cfg["output_dir"], "probe", args.model, stamp)
    print(f"→ output: {out_path}", file=sys.stderr)

    sem = asyncio.Semaphore(args.concurrency or cfg["concurrency"])
    done = 0

    async with httpx.AsyncClient() as gen_client, httpx.AsyncClient() as judge_client:
        async def worker(q):
            async with sem:
                rec = await score_one(backend, judge_client, cfg, judge_model, judge_key, gen_client, q)
                nonlocal done
                done += 1
                if done % 5 == 0 or done == len(questions):
                    print(f"  [{done}/{len(questions)}]", file=sys.stderr)
                return rec

        results = await asyncio.gather(*(worker(q) for q in questions))

    # Aggregate
    ok = [r for r in results if r.get("ok")]
    n_correct = sum(1 for r in ok if r.get("correct"))
    overall = round(n_correct / len(ok), 3) if ok else 0.0

    by_subject: dict[str, dict] = {}
    for r in ok:
        s = by_subject.setdefault(r["subject"], {"n": 0, "correct": 0})
        s["n"] += 1
        s["correct"] += int(bool(r["correct"]))
    for s in by_subject.values():
        s["accuracy"] = round(s["correct"] / s["n"], 3) if s["n"] else 0.0

    by_level: dict[str, dict] = {}
    for r in ok:
        lv = by_level.setdefault(r["level"], {"n": 0, "correct": 0})
        lv["n"] += 1
        lv["correct"] += int(bool(r["correct"]))
    for lv in by_level.values():
        lv["accuracy"] = round(lv["correct"] / lv["n"], 3) if lv["n"] else 0.0

    wrong = [
        {
            "id": r["id"],
            "subject": r["subject"],
            "level": r["level"],
            "question_he": r["question_he"],
            "gold_answer": r["gold_answer"],
            "answer": r["answer"],
            "judge_explanation": (r.get("judge") or {}).get("explanation", ""),
        }
        for r in ok if not r["correct"]
    ]

    payload = {
        "meta": {
            "kind": "probe",
            "model": args.model,
            "provider": args.provider,
            "base_url": args.base_url,
            "judge_model": judge_model,
            "pilot": args.pilot,
            "subjects": args.subjects,
            "timestamp": stamp,
            "n_questions": len(questions),
            "n_ok": len(ok),
            "n_failed": len(results) - len(ok),
        },
        "summary": {
            "accuracy": overall,
            "n_correct": n_correct,
            "n_scored": len(ok),
            "by_subject": by_subject,
            "by_level": by_level,
        },
        "wrong_answers": wrong,
        "results": results,
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    latest_symlink(out_path)

    print(f"\n✓ wrote {out_path}")
    print(f"  overall accuracy: {overall:.1%}  ({n_correct}/{len(ok)})")
    if by_level:
        print("  by level:")
        for lv in sorted(by_level):
            s = by_level[lv]
            print(f"    {lv:12s} {s['accuracy']:.1%}  ({s['correct']}/{s['n']})")
    if by_subject:
        print("  by subject:")
        for subj in sorted(by_subject):
            s = by_subject[subj]
            print(f"    {subj:22s} {s['accuracy']:.1%}  ({s['correct']}/{s['n']})")
    if wrong:
        print(f"\n  {len(wrong)} wrong (showing first 5):")
        for w in wrong[:5]:
            ans = (w["answer"] or "").replace("\n", " ")[:100]
            print(f"    [{w['id']:6s}] gold={w['gold_answer']!r}")
            print(f"             got: {ans}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic")
    p.add_argument("--base-url", help="OpenAI-compatible endpoint (vLLM etc.)")
    p.add_argument("--api-key", help="Override API key")
    p.add_argument("--judge-model", help="Override judge model (default: cfg.judge.model)")
    p.add_argument("--pilot", type=int, help="Run only first N questions")
    p.add_argument("--subjects", help="Comma-separated subject filter")
    p.add_argument("--concurrency", type=int)
    p.add_argument("--config", help="Path to config JSON")
    args = p.parse_args()

    cfg = load_config(args.config)
    asyncio.run(main_async(args, cfg))


if __name__ == "__main__":
    main()
