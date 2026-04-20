#!/usr/bin/env python3
"""
Generate model answers for RabBench questions.

Supports:
  - Anthropic direct API (sk-ant-api...)
  - OpenAI-compatible endpoints (OpenAI, vLLM, etc.) via --base-url
  - Local base models via --base-model (uses /v1/completions, no chat template)

Examples:
  python3 generate.py --model claude-sonnet-4-6 --pilot 20
  python3 generate.py --model claude-sonnet-4-6                    # full 825
  python3 generate.py --model gpt-4o --provider openai --pilot 20
  python3 generate.py --model Qwen3.5-9B-Base --provider openai \\
      --base-url http://localhost:8000/v1 --base-model --pilot 20
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx

from rabbench_io import (
    load_config,
    load_api_key,
    load_questions,
    result_path,
    latest_symlink,
    utc_stamp,
)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_user_prompt(q: dict) -> str:
    header_parts = []
    if q.get("subject_he"):
        header_parts.append(f"נושא: {q['subject_he']}")
    if q.get("level"):
        header_parts.append(f"רמה: {q['level']}")
    if q.get("section"):
        header_parts.append(f"סוג: {q['section']}")
    header = " | ".join(header_parts)

    return f"""שאלה ({header}):

{q['text']}

ענה באופן מלא ומנומק בעברית. ציין מקורות מדויקים. אם השאלה מחולקת לסעיפים (א/ב/ג/...), ענה על כל סעיף בנפרד."""


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

class AnthropicBackend:
    """Direct Anthropic Messages API."""

    def __init__(self, model: str, api_key: str, cfg: dict):
        self.model = model
        self.api_key = api_key
        self.url = cfg["api"]["anthropic"]["base_url"]
        self.max_tokens = cfg["generation_defaults"]["max_tokens"]
        self.temperature = cfg["generation_defaults"]["temperature"]
        self.system = cfg["generation_defaults"]["system_prompt_he"]

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
            "system": self.system,
            "messages": [{"role": "user", "content": build_user_prompt(q)}],
        }
        t0 = time.time()
        r = await client.post(self.url, headers=headers, json=payload, timeout=300.0)
        r.raise_for_status()
        data = r.json()
        text = "".join(blk.get("text", "") for blk in data.get("content", []) if blk.get("type") == "text")
        usage = data.get("usage", {})
        return {
            "answer": text,
            "latency_s": round(time.time() - t0, 2),
            "usage": {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
            },
            "raw_stop_reason": data.get("stop_reason"),
        }


class OpenAICompatibleBackend:
    """OpenAI /chat/completions or /completions (for --base-model)."""

    def __init__(self, model: str, api_key: str, base_url: str, cfg: dict, base_model: bool = False):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.max_tokens = cfg["generation_defaults"]["max_tokens"]
        self.temperature = cfg["generation_defaults"]["temperature"]
        self.system = cfg["generation_defaults"]["system_prompt_he"]
        self.base_model = base_model

    async def generate(self, client: httpx.AsyncClient, q: dict) -> dict:
        headers = {
            "authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
        }
        t0 = time.time()

        if self.base_model:
            # Raw completions — no chat template, minimal prompt
            prompt = f"{self.system}\n\nשאלה:\n{q['text']}\n\nתשובה:\n"
            payload = {
                "model": self.model,
                "prompt": prompt,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            }
            r = await client.post(f"{self.base_url}/completions", headers=headers, json=payload, timeout=900.0)
            r.raise_for_status()
            data = r.json()
            text = data["choices"][0].get("text", "")
        else:
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self.system},
                    {"role": "user", "content": build_user_prompt(q)},
                ],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            }
            r = await client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload, timeout=900.0)
            r.raise_for_status()
            data = r.json()
            text = data["choices"][0]["message"].get("content", "")

        usage = data.get("usage", {}) or {}
        return {
            "answer": text,
            "latency_s": round(time.time() - t0, 2),
            "usage": {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
            },
            "raw_stop_reason": (data.get("choices", [{}])[0].get("finish_reason")),
        }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def run_question(backend, client, q, retries: int = 2) -> dict:
    last_err = None
    for attempt in range(retries + 1):
        try:
            result = await backend.generate(client, q)
            return {
                "id": q["id"],
                "subject_en": q["subject_en"],
                "subject_he": q["subject_he"],
                "level": q["level"],
                "section": q["section"],
                "sub_questions": q.get("sub_questions", 1),
                "ok": True,
                **result,
            }
        except httpx.HTTPStatusError as e:
            last_err = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
        if attempt < retries:
            await asyncio.sleep(2 ** attempt)
    return {
        "id": q["id"],
        "subject_en": q["subject_en"],
        "ok": False,
        "error": last_err,
    }


async def main_async(args, cfg):
    meta, questions = load_questions(
        cfg["benchmark_file"],
        pilot=args.pilot,
        subjects=args.subjects.split(",") if args.subjects else None,
    )
    print(f"→ {len(questions)} questions loaded", file=sys.stderr)

    # Build backend
    if args.provider == "anthropic":
        key = load_api_key("anthropic", cfg)
        backend = AnthropicBackend(args.model, key, cfg)
    elif args.provider == "openai":
        key = load_api_key("openai", cfg) if not args.base_url else (args.api_key or "EMPTY")
        base_url = args.base_url or cfg["api"]["openai"]["base_url"]
        backend = OpenAICompatibleBackend(args.model, key, base_url, cfg, base_model=args.base_model)
    else:
        raise ValueError(f"Unknown provider: {args.provider}")

    stamp = utc_stamp()
    out_path = result_path(cfg["output_dir"], "gen", args.model, stamp)
    print(f"→ output: {out_path}", file=sys.stderr)

    sem = asyncio.Semaphore(args.concurrency or cfg["concurrency"])
    answers: list[dict] = []
    done = 0
    failed = 0

    async with httpx.AsyncClient() as client:
        async def worker(q):
            async with sem:
                result = await run_question(backend, client, q)
                nonlocal done, failed
                done += 1
                if not result["ok"]:
                    failed += 1
                if done % 5 == 0 or done == len(questions):
                    print(f"  [{done}/{len(questions)}] failed={failed}", file=sys.stderr)
                return result

        answers = await asyncio.gather(*(worker(q) for q in questions))

    # Aggregate usage
    total_in = sum(a.get("usage", {}).get("input_tokens", 0) for a in answers if a.get("ok"))
    total_out = sum(a.get("usage", {}).get("output_tokens", 0) for a in answers if a.get("ok"))

    payload = {
        "meta": {
            "kind": "generate",
            "model": args.model,
            "provider": args.provider,
            "base_url": args.base_url,
            "base_model": args.base_model,
            "pilot": args.pilot,
            "subjects": args.subjects,
            "timestamp": stamp,
            "n_questions": len(questions),
            "n_ok": len(questions) - failed,
            "n_failed": failed,
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
        },
        "benchmark_meta": meta,
        "answers": answers,
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    latest_symlink(out_path)
    print(f"✓ wrote {out_path} ({failed} failed, in={total_in} out={total_out})", file=sys.stderr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic")
    p.add_argument("--base-url", help="OpenAI-compatible endpoint (for local vLLM etc.)")
    p.add_argument("--base-model", action="store_true", help="Use /v1/completions (no chat template)")
    p.add_argument("--api-key", help="Override API key")
    p.add_argument("--pilot", type=int, help="Run only first N questions")
    p.add_argument("--subjects", help="Comma-separated subject_en filter")
    p.add_argument("--concurrency", type=int)
    p.add_argument("--config", help="Path to config JSON")
    args = p.parse_args()

    cfg = load_config(args.config)
    asyncio.run(main_async(args, cfg))


if __name__ == "__main__":
    main()
