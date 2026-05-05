#!/usr/bin/env python3
"""Autoresearch harness for Bonsai 1.7B and 8B models.

Runs the `BonsaiBenchmark/bonsaiEndToEndBenchmark` swift test across a set
of decode/prefill env-var knob combinations and ranks results by decode
median tok/s. Uses the active checkout (no worktree) for speed and to
avoid duplicate model loads.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Experiment:
    name: str
    env: dict[str, str]


@dataclass
class Result:
    name: str
    success: bool
    exit_code: int
    duration_s: float
    decode_median: float = 0.0
    decode_mean: float = 0.0
    decode_min: float = 0.0
    decode_max: float = 0.0
    decode_stddev: float = 0.0
    ttft_median_ms: float = 0.0
    greedy_prefix: list[int] = field(default_factory=list)
    error: str = ""
    report_path: str = ""


KNOBS: tuple[tuple[str, str], ...] = (
    ("EDGERUNNER_DECODE_FORCE_BASE", "decode_base"),
    ("EDGERUNNER_DECODE_DISABLE_MEGA_GQA", "no_mega_gqa"),
    ("EDGERUNNER_DECODE_DISABLE_FUSED_FINAL_NORM_LM_HEAD", "no_fused_final"),
    ("EDGERUNNER_DECODE_DISABLE_KV_BARRIER", "no_kv_barrier"),
    ("EDGERUNNER_DECODE_PREFER_METAL4", "metal4_decode"),
    ("EDGERUNNER_PREFILL_FORCE_LEGACY", "prefill_legacy"),
    ("EDGERUNNER_PREFILL_PREFER_METAL4", "metal4_prefill"),
)


def generate_experiments(limit: int) -> list[Experiment]:
    experiments: list[Experiment] = [Experiment("baseline", {})]
    seen = {"baseline"}
    for r in range(1, len(KNOBS) + 1):
        for combo in itertools.combinations(KNOBS, r):
            env = {k: "1" for k, _ in combo}
            name = "__".join(label for _, label in combo)
            if name in seen:
                continue
            experiments.append(Experiment(name, env))
            seen.add(name)
            if len(experiments) >= limit:
                return experiments
    return experiments


def run_experiment(
    *,
    repo_root: Path,
    artifacts_dir: Path,
    experiment: Experiment,
    model_path: Path,
    runs: int,
    tokens: int,
    context: int,
) -> Result:
    report_path = artifacts_dir / f"{experiment.name}.json"
    stdout_path = artifacts_dir / f"{experiment.name}.stdout.log"
    stderr_path = artifacts_dir / f"{experiment.name}.stderr.log"

    env = os.environ.copy()
    env.update(experiment.env)
    env["EDGERUNNER_BONSAI_MODEL_PATH"] = str(model_path)
    env["EDGERUNNER_BONSAI_OUTPUT_JSON"] = str(report_path)
    env["EDGERUNNER_BONSAI_RUNS"] = str(runs)
    env["EDGERUNNER_BONSAI_TOKENS"] = str(tokens)
    env["EDGERUNNER_BONSAI_CONTEXT"] = str(context)

    command = [
        "swift", "test", "-c", "release", "--skip-build",
        "--filter", "BonsaiBenchmark/bonsaiEndToEndBenchmark",
    ]
    start = time.monotonic()
    with stdout_path.open("w") as out, stderr_path.open("w") as err:
        completed = subprocess.run(
            command, cwd=repo_root, env=env, stdout=out, stderr=err, check=False
        )
    duration = time.monotonic() - start

    if completed.returncode != 0 or not report_path.exists():
        err_text = ""
        if stderr_path.exists():
            err_text = "\n".join(stderr_path.read_text(errors="replace").splitlines()[-10:])
        return Result(
            name=experiment.name,
            success=False,
            exit_code=completed.returncode,
            duration_s=duration,
            error=err_text or "no JSON report",
            report_path=str(report_path),
        )

    payload = json.loads(report_path.read_text())
    d = payload["decode_throughput"]
    return Result(
        name=experiment.name,
        success=True,
        exit_code=0,
        duration_s=duration,
        decode_median=float(d["median"]),
        decode_mean=float(d["mean"]),
        decode_min=float(d["min"]),
        decode_max=float(d["max"]),
        decode_stddev=float(d["stddev"]),
        ttft_median_ms=float(payload["ttft_ms"]["median"]),
        greedy_prefix=list(payload.get("greedy_prefix", [])),
        report_path=str(report_path),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to Bonsai GGUF",
    )
    parser.add_argument("--label", required=True, help="Short label (e.g. 1p7b, 8b)")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--reference-prefix",
        type=str,
        default="",
        help="Comma-separated greedy prefix IDs to enforce correctness",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    model_path = args.model.resolve()
    if not model_path.exists():
        print(f"model not found: {model_path}")
        return 2

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifacts_dir = (
        args.artifacts_dir.resolve()
        if args.artifacts_dir
        else repo_root / "benchmarks" / "autoresearch_runs" / f"bonsai_{args.label}_{ts}"
    )
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    print(f"artifacts: {artifacts_dir}")

    reference_prefix: Optional[list[int]] = None
    if args.reference_prefix:
        reference_prefix = [int(x) for x in args.reference_prefix.split(",") if x]

    experiments = generate_experiments(args.count)
    print(f"Queued {len(experiments)} experiments for Bonsai {args.label}")

    # Build first
    build = subprocess.run(
        ["swift", "build", "-c", "release"], cwd=repo_root, check=False
    )
    if build.returncode != 0:
        print("release build failed")
        return build.returncode

    all_results: list[Result] = []
    baseline: Optional[Result] = None
    for idx, exp in enumerate(experiments, start=1):
        r = run_experiment(
            repo_root=repo_root,
            artifacts_dir=artifacts_dir,
            experiment=exp,
            model_path=model_path,
            runs=args.runs,
            tokens=args.tokens,
            context=args.context,
        )
        if reference_prefix and r.success:
            if r.greedy_prefix[: len(reference_prefix)] != reference_prefix:
                r.success = False
                r.error = "greedy prefix mismatch"
        if exp.name == "baseline" and r.success:
            baseline = r
        all_results.append(r)
        status = "PASS" if r.success else "FAIL"
        print(
            f"[{idx}/{len(experiments)}] {status} {exp.name}: "
            f"{r.decode_median:.1f} tok/s  ttft={r.ttft_median_ms:.1f}ms  "
            f"({r.duration_s:.0f}s)"
        )

    # Persist
    with (artifacts_dir / "results.jsonl").open("w") as f:
        for r in all_results:
            f.write(json.dumps(asdict(r), sort_keys=True))
            f.write("\n")

    # Summary
    success = [r for r in all_results if r.success]
    success.sort(key=lambda r: -r.decode_median)
    lines = [
        f"# Bonsai {args.label} Autoresearch Summary",
        "",
        f"- Model: `{model_path}`",
        f"- Timestamp: {ts}",
        f"- Experiments: {len(all_results)} ({len(success)} success)",
        f"- Runs per experiment: {args.runs}",
        f"- Tokens per run: {args.tokens}",
        "",
    ]
    if baseline:
        lines.append(
            f"## Baseline: **{baseline.decode_median:.1f} tok/s** "
            f"(ttft={baseline.ttft_median_ms:.1f}ms)"
        )
        lines.append("")

    lines.extend(["## Top 10 Successful Experiments", ""])
    for r in success[:10]:
        delta = ""
        if baseline and baseline.decode_median > 0:
            pct = (r.decode_median - baseline.decode_median) / baseline.decode_median * 100
            delta = f" ({pct:+.1f}% vs baseline)"
        lines.append(
            f"- `{r.name}`: {r.decode_median:.1f} tok/s"
            f"{delta}, ttft={r.ttft_median_ms:.1f}ms"
        )

    lines.extend(["", "## Failures", ""])
    failures = [r for r in all_results if not r.success]
    for r in failures:
        short = r.error.splitlines()[-1] if r.error else "unknown"
        lines.append(f"- `{r.name}`: {short[:120]}")

    (artifacts_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(f"\nWrote {artifacts_dir}/SUMMARY.md")
    if baseline:
        best = success[0] if success else baseline
        delta_pct = (best.decode_median - baseline.decode_median) / baseline.decode_median * 100
        print(
            f"Best: {best.name} = {best.decode_median:.1f} tok/s "
            f"({delta_pct:+.1f}% vs baseline {baseline.decode_median:.1f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
