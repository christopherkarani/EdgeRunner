#!/usr/bin/env python3
import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
from transformers import AutoTokenizer


REPO_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_DIR / "benchmarks" / "long_prompt_framework_comparison.json"
DEFAULT_PROMPT_REPO = "Qwen/Qwen3-0.6B"
DEFAULT_MLX_REPO = "mlx-community/Qwen3-0.6B-8bit"


def build_prompt_text() -> str:
    paragraph = textwrap.dedent(
        """
        You are reviewing a long systems design brief for an on-device language model runtime.
        Summarize the performance constraints, memory tradeoffs, kernel architecture choices,
        benchmark methodology, and deployment risks. Keep the analysis factual and specific.

        Section:
        The runtime uses paged KV cache growth, aggressive fused kernels for decode, and a
        separate prefill path that must remain correct for long contexts. The team cares about
        prompt processing throughput, first-token latency, and decode throughput after the cache
        is populated. They also care about deterministic output on a pinned artifact, explicit
        benchmark contracts, and avoiding misleading tiny-prompt numbers.
        """
    ).strip()
    return "\n\n".join(paragraph for _ in range(256))


def exact_prompt_tokens(tokenizer, target_tokens: int) -> list[int]:
    base_text = build_prompt_text()
    token_ids = tokenizer.encode(base_text, add_special_tokens=True)
    if len(token_ids) < target_tokens:
        raise RuntimeError(
            f"Generated prompt only produced {len(token_ids)} tokens, need at least {target_tokens}"
        )
    return token_ids[:target_tokens]


def median(values: list[float]) -> float:
    return statistics.median(values)


def mean(values: list[float]) -> float:
    return statistics.fmean(values)


def stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def aggregate_runs(framework: str, runs: list[dict], metadata: dict) -> dict:
    prompt_tok_s = [run["prompt_tok_s"] for run in runs]
    ttft_ms = [run["ttft_ms"] for run in runs]
    decode_tok_s = [run["decode_tok_s"] for run in runs]
    return {
        "framework": framework,
        "runs": runs,
        "summary": {
            "prompt_tok_s": {
                "median": median(prompt_tok_s),
                "mean": mean(prompt_tok_s),
                "stdev": stdev(prompt_tok_s),
                "min": min(prompt_tok_s),
                "max": max(prompt_tok_s),
            },
            "ttft_ms": {
                "median": median(ttft_ms),
                "mean": mean(ttft_ms),
                "stdev": stdev(ttft_ms),
                "min": min(ttft_ms),
                "max": max(ttft_ms),
            },
            "decode_tok_s": {
                "median": median(decode_tok_s),
                "mean": mean(decode_tok_s),
                "stdev": stdev(decode_tok_s),
                "min": min(decode_tok_s),
                "max": max(decode_tok_s),
            },
        },
        "metadata": metadata,
    }


def run_edge_runner_child(prompt_path: Path, output_path: Path, generate_count: int, context_window: int) -> dict:
    env = os.environ.copy()
    env.update(
        {
            "EDGERUNNER_LONG_PROMPT_CHILD_RUN": "1",
            "EDGERUNNER_LONG_PROMPT_TOKENS_PATH": str(prompt_path),
            "EDGERUNNER_LONG_PROMPT_OUTPUT_PATH": str(output_path),
            "EDGERUNNER_LONG_PROMPT_GENERATE_COUNT": str(generate_count),
            "EDGERUNNER_LONG_PROMPT_CONTEXT_WINDOW": str(context_window),
        }
    )
    cmd = [
        "swift",
        "test",
        "-c",
        "release",
        "--filter",
        "LongPromptFrameworkBenchmark/edgeRunnerChildRun",
    ]
    subprocess.run(cmd, cwd=REPO_DIR, env=env, check=True)
    return json.loads(output_path.read_text())


def run_mlx_child(prompt_tokens: list[int], mlx_repo: str, generate_count: int) -> dict:
    model, _tokenizer = load(mlx_repo)
    prompt = mx.array(prompt_tokens)

    # Warm the exact prompt shape once so the measured pass reflects steady-state inference.
    warmup = generate_step(prompt, model, max_tokens=1)
    next(warmup)
    mx.clear_cache()

    generator = generate_step(prompt, model, max_tokens=generate_count)

    ttft_start = time.perf_counter()
    first_token, _first_logprobs = next(generator)
    mx.eval(first_token)
    ttft_seconds = time.perf_counter() - ttft_start

    decode_start = time.perf_counter()
    remaining = 0
    for remaining, (_token, _logprobs) in enumerate(generator, start=1):
        pass
    decode_seconds = time.perf_counter() - decode_start

    measured_decode_count = remaining
    decode_tok_s = measured_decode_count / decode_seconds if measured_decode_count > 0 else 0.0

    return {
        "framework": "MLX",
        "prompt_token_count": len(prompt_tokens),
        "generated_token_count": generate_count,
        "ttft_ms": ttft_seconds * 1000,
        "prompt_tok_s": len(prompt_tokens) / ttft_seconds,
        "decode_tok_s": decode_tok_s,
        "peak_memory_gb": float(mx.get_peak_memory() / 1e9),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-tokens", type=int, default=1024)
    parser.add_argument("--generate-tokens", type=int, default=128)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--context-window", type=int, default=2048)
    parser.add_argument("--prompt-tokenizer-repo", default=DEFAULT_PROMPT_REPO)
    parser.add_argument("--mlx-repo", default=DEFAULT_MLX_REPO)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.prompt_tokens + args.generate_tokens > args.context_window:
        raise SystemExit("prompt-tokens + generate-tokens must fit inside context-window")

    tokenizer = AutoTokenizer.from_pretrained(args.prompt_tokenizer_repo)
    prompt_tokens = exact_prompt_tokens(tokenizer, args.prompt_tokens)

    with tempfile.TemporaryDirectory(prefix="edgerunner-long-prompt-bench-") as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        prompt_path = tmp_dir_path / "prompt_tokens.json"
        prompt_path.write_text(json.dumps({"prompt_tokens": prompt_tokens}))

        edge_runner_runs = []
        mlx_runs = []

        for run_index in range(args.runs):
            edge_output_path = tmp_dir_path / f"edge_runner_run_{run_index}.json"
            edge_runner_runs.append(
                run_edge_runner_child(
                    prompt_path=prompt_path,
                    output_path=edge_output_path,
                    generate_count=args.generate_tokens,
                    context_window=args.context_window,
                )
            )
            mlx_runs.append(
                run_mlx_child(
                    prompt_tokens=prompt_tokens,
                    mlx_repo=args.mlx_repo,
                    generate_count=args.generate_tokens,
                )
            )

    edge_summary = aggregate_runs(
        framework="EdgeRunner",
        runs=edge_runner_runs,
        metadata={
            "model_path": "/tmp/edgerunner-models/Qwen3-0.6B-Q8_0.gguf",
            "context_window": args.context_window,
        },
    )
    mlx_summary = aggregate_runs(
        framework="MLX",
        runs=mlx_runs,
        metadata={
            "model_repo": args.mlx_repo,
            "context_window": args.context_window,
        },
    )

    result = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "prompt_tokenizer_repo": args.prompt_tokenizer_repo,
        "prompt_token_count": len(prompt_tokens),
        "generate_token_count": args.generate_tokens,
        "runs": args.runs,
        "prompt_preview_tokens": prompt_tokens[:16],
        "frameworks": [edge_summary, mlx_summary],
        "delta": {
            "prompt_tok_s_pct_mlx_over_edgerunner": (
                (mlx_summary["summary"]["prompt_tok_s"]["median"] / edge_summary["summary"]["prompt_tok_s"]["median"]) - 1.0
            )
            * 100.0,
            "ttft_pct_edgerunner_faster_than_mlx": (
                1.0 - (edge_summary["summary"]["ttft_ms"]["median"] / mlx_summary["summary"]["ttft_ms"]["median"])
            )
            * 100.0,
            "decode_tok_s_pct_mlx_over_edgerunner": (
                (mlx_summary["summary"]["decode_tok_s"]["median"] / edge_summary["summary"]["decode_tok_s"]["median"]) - 1.0
            )
            * 100.0,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
