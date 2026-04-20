#!/usr/bin/env python3
"""
End-to-end RabBench eval for a trained model:
  1. Launch a vLLM OpenAI-compatible server (Docker) pointing at --model-dir
  2. Wait for /v1/models to come up
  3. Run bench/generate.py against the local endpoint
  4. Run bench/judge.py (Anthropic Opus judge by default)
  5. Run bench/compare.py against any provided baselines
  6. Tear down the Docker container on exit (including Ctrl-C / errors)

The Docker image used here (vllm-openai:qwen3_5-hardened) is the one that works
with Qwen3.5 on GB10 per dgx-spark-playbooks/nvidia/vllm notes.

Usage:
  python3 train/eval_rabbench.py \\
      --model-dir /home/aigroup/training/rabbench-sft/merged \\
      --model-name rabbench-sft-v1 \\
      --pilot 20 \\
      --baseline ../results/judge_Qwen3.5-9B-Base_latest.json \\
      --baseline ../results/judge_claude-sonnet-4-6_latest.json
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import urllib.request
import urllib.error


REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = REPO_ROOT / "bench"
RESULTS_DIR = REPO_ROOT / "results"

DEFAULT_IMAGE = "vllm-openai:qwen3_5-hardened"
DEFAULT_CONTAINER = "rabbench-eval-vllm"


def log(msg: str) -> None:
    print(f"[eval] {msg}", flush=True)


def vllm_ready(base_url: str, timeout_s: float = 3.0) -> bool:
    try:
        req = urllib.request.Request(f"{base_url}/models")
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            return r.status == 200
    except (urllib.error.URLError, urllib.error.HTTPError, OSError):
        return False


def start_vllm(args: argparse.Namespace) -> None:
    # Stop any prior container with the same name.
    subprocess.run(
        ["docker", "rm", "-f", args.container],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    docker_cmd = [
        "docker", "run", "-d",
        "--name", args.container,
        "--gpus", "all",
        "--ipc=host",
        "-v", f"{args.model_dir}:/model",
        "-p", f"{args.port}:8000",
        args.image,
        "--model", "/model",
        "--served-model-name", args.model_name,
        "--dtype", "bfloat16",
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_mem_util),
        "--trust-remote-code",
        "--enforce-eager",  # required on sm_121 Qwen3.5 (DeltaNet torch.compile crash)
        "--disable-log-requests",
    ]
    log("Launching vLLM container:")
    log("  " + " ".join(docker_cmd))
    proc = subprocess.run(docker_cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit(f"docker run failed:\n{proc.stderr}")
    log(f"Container started: {proc.stdout.strip()[:12]}")


def stop_vllm(container: str) -> None:
    log(f"Stopping container {container}...")
    subprocess.run(["docker", "rm", "-f", container], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def wait_for_vllm(base_url: str, container: str, timeout_s: int) -> None:
    log(f"Waiting for vLLM at {base_url} (up to {timeout_s}s)...")
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if vllm_ready(base_url):
            log(f"  vLLM ready after {time.time() - t0:.0f}s")
            return
        # Check the container is still alive.
        inspect = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", container],
            capture_output=True, text=True,
        )
        if inspect.returncode != 0 or inspect.stdout.strip() != "true":
            logs = subprocess.run(
                ["docker", "logs", "--tail", "80", container],
                capture_output=True, text=True,
            ).stdout
            raise RuntimeError(f"vLLM container died during startup.\n--- last logs ---\n{logs}")
        time.sleep(5)
    raise TimeoutError(f"vLLM did not come up within {timeout_s}s")


def run(cmd: list[str], cwd: Path | None = None) -> None:
    log("$ " + " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed (exit {proc.returncode}): {' '.join(cmd)}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", required=True, help="Merged bf16 model directory")
    p.add_argument("--model-name", default="rabbench-sft", help="Name to report to bench scripts + vLLM")
    p.add_argument("--pilot", type=int, default=20, help="Number of questions (None/0 = full 825)")
    p.add_argument("--subjects", help="Comma-separated subject_en filter")
    p.add_argument("--baseline", action="append", default=[], help="Prior judge_*.json to compare against (repeatable)")

    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--container", default=DEFAULT_CONTAINER)
    p.add_argument("--image", default=DEFAULT_IMAGE)
    p.add_argument("--gpu-mem-util", type=float, default=0.85)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--startup-timeout", type=int, default=600)

    p.add_argument("--judge-model", help="Override judge model (default: configs/default.json)")
    p.add_argument("--skip-serve", action="store_true", help="Assume vLLM is already running on --port")
    p.add_argument("--keep-serving", action="store_true", help="Do not shut down the container after eval")
    args = p.parse_args()

    model_dir = Path(args.model_dir).resolve()
    if not model_dir.is_dir():
        sys.exit(f"--model-dir not found: {model_dir}")

    base_url = f"http://localhost:{args.port}/v1"

    # Start vLLM and ensure we tear it down on any exit path.
    started_here = False
    try:
        if not args.skip_serve:
            start_vllm(args)
            started_here = True
            wait_for_vllm(base_url, args.container, args.startup_timeout)
        else:
            if not vllm_ready(base_url):
                sys.exit(f"--skip-serve was set but {base_url} is not responding")

        # 1. Generate
        gen_cmd = [
            sys.executable, str(BENCH_DIR / "generate.py"),
            "--model", args.model_name,
            "--provider", "openai",
            "--base-url", base_url,
        ]
        if args.pilot:
            gen_cmd += ["--pilot", str(args.pilot)]
        if args.subjects:
            gen_cmd += ["--subjects", args.subjects]
        run(gen_cmd, cwd=BENCH_DIR)

        gen_latest = RESULTS_DIR / f"gen_{_slug(args.model_name)}_latest.json"
        if not gen_latest.exists():
            raise RuntimeError(f"expected {gen_latest} after generate.py")

        # 2. Judge
        judge_cmd = [
            sys.executable, str(BENCH_DIR / "judge.py"),
            "--input", str(gen_latest),
        ]
        if args.judge_model:
            judge_cmd += ["--judge-model", args.judge_model]
        run(judge_cmd, cwd=BENCH_DIR)

        judge_latest = RESULTS_DIR / f"judge_{_slug(args.model_name)}_latest.json"
        if not judge_latest.exists():
            raise RuntimeError(f"expected {judge_latest} after judge.py")

        # 3. Compare
        compare_cmd = [sys.executable, str(BENCH_DIR / "compare.py"), str(judge_latest)]
        for b in args.baseline:
            bp = Path(b).resolve()
            if not bp.exists():
                log(f"  warning: baseline {bp} missing, skipping")
                continue
            compare_cmd.append(str(bp))
        run(compare_cmd, cwd=BENCH_DIR)

    finally:
        if started_here and not args.keep_serving:
            stop_vllm(args.container)


def _slug(name: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-")


if __name__ == "__main__":
    # Make sure Ctrl-C tears the container down cleanly.
    def _handler(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)

    try:
        main()
    except KeyboardInterrupt:
        log("interrupted")
        subprocess.run(
            ["docker", "rm", "-f", DEFAULT_CONTAINER],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        sys.exit(130)
