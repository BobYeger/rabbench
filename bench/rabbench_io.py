"""Shared I/O helpers: config, api keys, question loading, result persistence."""

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

BENCH_DIR = Path(__file__).parent
REPO_ROOT = BENCH_DIR.parent


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else BENCH_DIR / "configs" / "default.json"
    with open(path) as f:
        cfg = json.load(f)
    # Resolve paths relative to bench/
    cfg["benchmark_file"] = str((BENCH_DIR / cfg["benchmark_file"]).resolve())
    cfg["rubric_file"] = str((BENCH_DIR / cfg["rubric_file"]).resolve())
    cfg["output_dir"] = str((BENCH_DIR / cfg["output_dir"]).resolve())
    return cfg


def load_api_key(provider: str, cfg: dict) -> str:
    """Resolve API key for a provider. Env var wins; otherwise parse the key file."""
    api_cfg = cfg["api"][provider]
    env_var = api_cfg.get("key_env")
    if env_var:
        key = os.environ.get(env_var, "").strip()
        if key:
            return key

    key_file = api_cfg.get("key_file")
    if key_file:
        path = Path(os.path.expanduser(key_file))
        if path.exists():
            text = path.read_text()
            marker = api_cfg.get("key_marker", provider)
            # Find "## <marker>" section and extract the "API key: ..." line.
            sections = re.split(r"^##\s+", text, flags=re.MULTILINE)
            for section in sections:
                if section.startswith(marker):
                    m = re.search(r"API key:\s*(\S+)", section)
                    if m:
                        return m.group(1).strip()
            # Fallback: first matching API key line after marker
            m = re.search(
                rf"{re.escape(marker)}.*?API key:\s*(\S+)",
                text,
                re.DOTALL,
            )
            if m:
                return m.group(1).strip()

    raise RuntimeError(f"No API key found for provider '{provider}'")


def load_questions(path: str, pilot: int | None = None, subjects: list[str] | None = None, ids: list[str] | None = None) -> tuple[dict, list[dict]]:
    """Return (metadata, questions). Deterministic ordering: by id."""
    with open(path) as f:
        data = json.load(f)
    questions = data["questions"]
    if subjects:
        questions = [q for q in questions if q["subject_en"] in subjects]
    if ids:
        idset = set(ids)
        questions = [q for q in questions if q["id"] in idset]
    questions = sorted(questions, key=lambda q: q["id"])
    if pilot is not None:
        questions = questions[:pilot]
    return data.get("metadata", {}), questions


def load_rubric(path: str) -> str:
    with open(path) as f:
        return f.read()


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def slug_model(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", model).strip("-")


def result_path(output_dir: str, kind: str, model: str, stamp: str | None = None) -> Path:
    """kind = 'gen' | 'judge' | 'compare'"""
    stamp = stamp or utc_stamp()
    fname = f"{kind}_{slug_model(model)}_{stamp}.json"
    path = Path(output_dir) / fname
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def latest_symlink(path: Path) -> None:
    """Update a <kind>_<model>_latest.json symlink alongside the real file."""
    target = path.name
    link = path.parent / re.sub(r"_\d{8}T\d{6}Z\.json$", "_latest.json", path.name)
    try:
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(target)
    except OSError:
        pass  # non-fatal


def save_jsonl_streaming(path: Path):
    """Context manager that appends JSON lines; converts to .json at the end."""
    raise NotImplementedError  # placeholder for future streaming writer
