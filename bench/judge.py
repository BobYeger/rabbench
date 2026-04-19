#!/usr/bin/env python3
"""
Judge model answers against the RabBench rubric using Opus 4.6 (or other judge).

Usage:
  python3 judge.py --input ../results/gen_claude-sonnet-4-6_latest.json
  python3 judge.py --input ../results/gen_claude-sonnet-4-6_latest.json --pilot 20
  python3 judge.py --input ... --judge-model claude-opus-4-6
"""

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

import httpx

from rabbench_io import (
    load_config,
    load_api_key,
    load_rubric,
    result_path,
    latest_symlink,
    utc_stamp,
)


JUDGE_SYSTEM_PROMPT = """You are an expert judge of rabbinic scholarship answering a halachic essay question. You grade strictly against a provided rubric, returning JSON only. You read Hebrew fluently. You are not the respondent; do not answer the question yourself."""


def build_judge_prompt(rubric: str, q_text: str, subject_he: str, level: str, sub_questions: int, answer: str) -> str:
    return f"""You will grade an answer to a rabbinic exam question.

# RUBRIC
{rubric}

# TASK

**Subject (Hebrew):** {subject_he}
**Level:** {level}
**Sub-questions:** {sub_questions}

**Question:**
```
{q_text}
```

**Candidate answer:**
```
{answer}
```

# OUTPUT

Respond with a single JSON object and nothing else. Do not wrap in markdown fences. Schema:

{{
  "accuracy":     <int 1-5>,
  "sources":      <int 1-5>,
  "reasoning":    <int 1-5>,
  "completeness": <int 1-5>,
  "language":     <int 1-5>,
  "rationale_he": "<short Hebrew explanation, 2-4 sentences, referencing the specific grading>",
  "red_flags":    ["<zero or more of: fabricated_citation, wrong_attribution, refused_to_answer, off_topic, english_only, incomplete>"]
}}

Use 5 only when the dimension is truly excellent. Use 1 when it is fundamentally failing. Be strict. If the answer is in the wrong language (e.g., English), that's `english_only` and language score <= 2."""


JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_judge_output(text: str) -> dict | None:
    text = text.strip()
    # Strip potential ```json fences defensively
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


async def judge_one(client, cfg, api_key, judge_model, rubric, ans_record) -> dict:
    if not ans_record.get("ok"):
        return {
            "id": ans_record["id"],
            "ok": False,
            "error": f"generation failed: {ans_record.get('error')}",
        }

    payload = {
        "model": judge_model,
        "max_tokens": cfg["judge"]["max_tokens"],
        "temperature": cfg["judge"]["temperature"],
        "system": JUDGE_SYSTEM_PROMPT,
        "messages": [{
            "role": "user",
            "content": build_judge_prompt(
                rubric=rubric,
                q_text=ans_record["_q_text"],
                subject_he=ans_record.get("subject_he", ""),
                level=ans_record.get("level", ""),
                sub_questions=ans_record.get("sub_questions", 1),
                answer=ans_record.get("answer", ""),
            ),
        }],
    }
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    url = cfg["api"]["anthropic"]["base_url"]

    last_err = None
    for attempt in range(3):
        try:
            t0 = time.time()
            r = await client.post(url, headers=headers, json=payload, timeout=300.0)
            r.raise_for_status()
            data = r.json()
            text = "".join(blk.get("text", "") for blk in data.get("content", []) if blk.get("type") == "text")
            parsed = parse_judge_output(text)
            if not parsed:
                last_err = f"unparseable judge output: {text[:200]}"
                continue
            usage = data.get("usage", {})
            return {
                "id": ans_record["id"],
                "subject_en": ans_record.get("subject_en"),
                "ok": True,
                "scores": {
                    "accuracy": int(parsed.get("accuracy", 0)),
                    "sources": int(parsed.get("sources", 0)),
                    "reasoning": int(parsed.get("reasoning", 0)),
                    "completeness": int(parsed.get("completeness", 0)),
                    "language": int(parsed.get("language", 0)),
                },
                "rationale_he": parsed.get("rationale_he", ""),
                "red_flags": parsed.get("red_flags", []),
                "judge_latency_s": round(time.time() - t0, 2),
                "judge_usage": {
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                },
            }
        except httpx.HTTPStatusError as e:
            last_err = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
        await asyncio.sleep(2 ** attempt)

    return {"id": ans_record["id"], "ok": False, "error": last_err}


def weighted_total(s: dict) -> float:
    return (
        1.0 * s["accuracy"]
        + 1.0 * s["sources"]
        + 1.0 * s["reasoning"]
        + 1.0 * s["completeness"]
        + 0.5 * s["language"]
    )


def summarize(judgments: list[dict]) -> dict:
    ok = [j for j in judgments if j.get("ok")]
    if not ok:
        return {"n": 0}
    dims = ["accuracy", "sources", "reasoning", "completeness", "language"]
    avg_per_dim = {d: round(sum(j["scores"][d] for j in ok) / len(ok), 2) for d in dims}
    raw_totals = [sum(j["scores"].values()) for j in ok]
    w_totals = [weighted_total(j["scores"]) for j in ok]

    # By subject
    by_subj: dict[str, list[dict]] = {}
    for j in ok:
        by_subj.setdefault(j.get("subject_en", "?"), []).append(j)
    subj_summary = {
        subj: {
            "n": len(items),
            "avg_raw": round(sum(sum(it["scores"].values()) for it in items) / len(items), 2),
            "avg_weighted": round(sum(weighted_total(it["scores"]) for it in items) / len(items), 2),
        }
        for subj, items in by_subj.items()
    }

    # Red flag counts
    flag_counts: dict[str, int] = {}
    for j in ok:
        for f in j.get("red_flags") or []:
            flag_counts[f] = flag_counts.get(f, 0) + 1

    return {
        "n": len(ok),
        "n_failed": len(judgments) - len(ok),
        "avg_per_dim": avg_per_dim,
        "avg_raw_total": round(sum(raw_totals) / len(raw_totals), 2),
        "avg_weighted_total": round(sum(w_totals) / len(w_totals), 2),
        "by_subject": subj_summary,
        "red_flag_counts": flag_counts,
    }


async def main_async(args, cfg):
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = Path.cwd() / input_path
    with open(input_path) as f:
        gen_data = json.load(f)

    # Merge in original question text by id (judge needs it)
    with open(cfg["benchmark_file"]) as f:
        bench = json.load(f)
    q_by_id = {q["id"]: q for q in bench["questions"]}

    answers = gen_data["answers"]
    for a in answers:
        q = q_by_id.get(a["id"])
        a["_q_text"] = q["text"] if q else ""

    if args.pilot:
        answers = answers[: args.pilot]

    rubric = load_rubric(cfg["rubric_file"])
    api_key = load_api_key("anthropic", cfg)
    judge_model = args.judge_model or cfg["judge"]["model"]

    stamp = utc_stamp()
    gen_model = gen_data["meta"]["model"]
    out_path = result_path(cfg["output_dir"], "judge", gen_model, stamp)
    print(f"→ judging {len(answers)} answers with {judge_model}", file=sys.stderr)
    print(f"→ output: {out_path}", file=sys.stderr)

    sem = asyncio.Semaphore(args.concurrency or cfg["concurrency"])
    done = 0

    async with httpx.AsyncClient() as client:
        async def worker(rec):
            async with sem:
                j = await judge_one(client, cfg, api_key, judge_model, rubric, rec)
                nonlocal done
                done += 1
                if done % 5 == 0 or done == len(answers):
                    print(f"  [{done}/{len(answers)}]", file=sys.stderr)
                return j

        judgments = await asyncio.gather(*(worker(a) for a in answers))

    summary = summarize(judgments)

    payload = {
        "meta": {
            "kind": "judge",
            "source_generate_file": str(input_path),
            "generation_model": gen_model,
            "judge_model": judge_model,
            "timestamp": stamp,
            "n_answers": len(answers),
            "n_scored": summary.get("n", 0),
            "n_failed": summary.get("n_failed", 0),
        },
        "summary": summary,
        "judgments": judgments,
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    latest_symlink(out_path)

    print(f"\n✓ wrote {out_path}")
    print(f"  avg weighted total: {summary.get('avg_weighted_total')} / 22.5")
    print(f"  avg per dim: {summary.get('avg_per_dim')}")
    if summary.get("red_flag_counts"):
        print(f"  red flags: {summary['red_flag_counts']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="generate.py output JSON")
    p.add_argument("--judge-model", help="Override judge model")
    p.add_argument("--pilot", type=int, help="Judge only first N")
    p.add_argument("--concurrency", type=int)
    p.add_argument("--config", help="Path to config JSON")
    args = p.parse_args()

    cfg = load_config(args.config)
    asyncio.run(main_async(args, cfg))


if __name__ == "__main__":
    main()
