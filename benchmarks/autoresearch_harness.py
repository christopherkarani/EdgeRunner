#!/usr/bin/env python3

from __future__ import annotations

import argparse
import itertools
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Experiment:
    name: str
    env: dict[str, str]


@dataclass(frozen=True)
class BenchmarkResult:
    decode_median: float
    decode_max: float
    decode_mean: float
    decode_stddev: float
    decode_min: float
    ttft_median_ms: float
    ttft_mean_ms: float
    model_load_mb: float
    peak_rss_mb: float
    deterministic: bool
    greedy_prefix: list[int]
    token_hash: str
    is_canonical_run: bool

    def matches_contract(self, *, expected_prefix: list[int], expected_hash: str) -> bool:
        return (
            self.deterministic
            and self.greedy_prefix[: len(expected_prefix)] == expected_prefix
            and self.token_hash == expected_hash
        )


@dataclass(frozen=True)
class BenchmarkContract:
    expected_prefix: list[int]
    expected_hash: str
    token_count: int
    run_count: int
    context_window: int


@dataclass(frozen=True)
class CompletedExperiment:
    name: str
    env: dict[str, str]
    success: bool
    exit_code: int
    duration_seconds: float
    report_path: str
    stdout_path: str
    stderr_path: str
    error: str | None
    result: BenchmarkResult | None


KNOBS: tuple[tuple[str, str], ...] = (
    ("EDGERUNNER_DECODE_FORCE_BASE", "decode_base"),
    ("EDGERUNNER_DECODE_DISABLE_MEGA_GQA", "decode_no_mega"),
    ("EDGERUNNER_DECODE_DISABLE_FUSED_FINAL_NORM_LM_HEAD", "decode_no_fused_final"),
    ("EDGERUNNER_DECODE_DISABLE_KV_BARRIER", "decode_no_kv_barrier"),
    ("EDGERUNNER_DECODE_PREFER_METAL4", "decode_metal4"),
    ("EDGERUNNER_PREFILL_FORCE_LEGACY", "prefill_legacy"),
    ("EDGERUNNER_PREFILL_PREFER_METAL4", "prefill_metal4"),
)


def generate_experiments(limit: int) -> list[Experiment]:
    if limit < 1:
        raise ValueError("limit must be >= 1")

    experiments = [Experiment(name="baseline", env={})]
    seen_names = {"baseline"}

    for active_count in range(1, len(KNOBS) + 1):
        for combo in itertools.combinations(KNOBS, active_count):
            env = {key: "1" for key, _label in combo}
            labels = [label for _key, label in combo]
            name = "__".join(labels)
            if name in seen_names:
                continue
            experiments.append(Experiment(name=name, env=env))
            seen_names.add(name)
            if len(experiments) >= limit:
                return experiments

    if len(experiments) < limit:
        raise ValueError(f"unable to generate {limit} unique experiments from {len(KNOBS)} knobs")
    return experiments


def load_contract(contract_path: Path) -> BenchmarkContract:
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    publishable = payload["publishable"]
    return BenchmarkContract(
        expected_prefix=list(publishable["expected_greedy_prefix"]),
        expected_hash=str(publishable["expected_token_hash"]),
        token_count=int(publishable["token_count"]),
        run_count=int(publishable["run_count"]),
        context_window=int(publishable["context_window"]),
    )


def load_benchmark_result(report_path: Path) -> BenchmarkResult:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    decode = payload["decode_throughput"]
    ttft = payload["ttft_ms"]
    memory = payload["memory_mb"]
    return BenchmarkResult(
        decode_median=float(decode["median"]),
        decode_max=float(decode["max"]),
        decode_mean=float(decode["mean"]),
        decode_stddev=float(decode["stddev"]),
        decode_min=float(decode["min"]),
        ttft_median_ms=float(ttft["median"]),
        ttft_mean_ms=float(ttft["mean"]),
        model_load_mb=float(memory["model_load"]),
        peak_rss_mb=float(memory["peak_rss"]),
        deterministic=bool(payload["deterministic"]),
        greedy_prefix=list(payload["greedy_prefix"]),
        token_hash=str(payload["token_hash"]),
        is_canonical_run=bool(payload["is_canonical_run"]),
    )


def rank_results(
    rows: Iterable[tuple[str, BenchmarkResult, bool]]
) -> list[tuple[str, BenchmarkResult, bool]]:
    return sorted(
        rows,
        key=lambda row: (
            not row[2],
            -row[1].decode_median,
            row[1].ttft_median_ms,
        ),
    )


def run_command(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
) -> tuple[int, float]:
    start = time.monotonic()
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_handle:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdout=stdout_handle,
            stderr=stderr_handle,
            check=False,
        )
    duration = time.monotonic() - start
    return completed.returncode, duration


def create_worktree(repo_root: Path, *, keep: bool) -> tuple[Path, callable]:
    base_dir = Path(tempfile.mkdtemp(prefix="edgerunner-autoresearch-"))
    worktree_path = base_dir / "worktree"
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(worktree_path), "HEAD"],
        cwd=repo_root,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    def cleanup() -> None:
        if keep:
            return
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(worktree_path)],
            cwd=repo_root,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        shutil.rmtree(base_dir, ignore_errors=True)

    return worktree_path, cleanup


def run_single_experiment(
    *,
    repo_root: Path,
    worktree_path: Path,
    artifacts_dir: Path,
    contract: BenchmarkContract,
    experiment: Experiment,
    run_count: int,
    context_window: int,
    token_count: int,
) -> CompletedExperiment:
    report_path = artifacts_dir / f"{experiment.name}.json"
    stdout_path = artifacts_dir / f"{experiment.name}.stdout.log"
    stderr_path = artifacts_dir / f"{experiment.name}.stderr.log"

    env = os.environ.copy()
    env.update(experiment.env)
    env["PROJECT_DIR"] = str(worktree_path)
    env["EDGERUNNER_BENCHMARK_TOKENS"] = str(token_count)
    env["EDGERUNNER_BENCHMARK_RUNS"] = str(run_count)
    env["EDGERUNNER_BENCHMARK_CONTEXT"] = str(context_window)
    env["EDGERUNNER_BENCHMARK_OUTPUT_PATH"] = str(report_path)

    command = [
        "swift",
        "test",
        "-c",
        "release",
        "--skip-build",
        "--filter",
        "PublishableBenchmark/fullBenchmark",
    ]
    exit_code, duration_seconds = run_command(
        command,
        cwd=worktree_path,
        env=env,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )

    if exit_code != 0:
        error = tail_text(stderr_path) or tail_text(stdout_path) or "benchmark command failed"
        return CompletedExperiment(
            name=experiment.name,
            env=experiment.env,
            success=False,
            exit_code=exit_code,
            duration_seconds=duration_seconds,
            report_path=str(report_path),
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            error=error,
            result=None,
        )

    if not report_path.exists():
        return CompletedExperiment(
            name=experiment.name,
            env=experiment.env,
            success=False,
            exit_code=exit_code,
            duration_seconds=duration_seconds,
            report_path=str(report_path),
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            error="benchmark completed without writing JSON report",
            result=None,
        )

    result = load_benchmark_result(report_path)
    success = result.matches_contract(
        expected_prefix=contract.expected_prefix,
        expected_hash=contract.expected_hash,
    )
    error = None if success else "correctness contract failed"
    return CompletedExperiment(
        name=experiment.name,
        env=experiment.env,
        success=success,
        exit_code=exit_code,
        duration_seconds=duration_seconds,
        report_path=str(report_path),
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        error=error,
        result=result,
    )


def tail_text(path: Path, *, line_count: int = 20) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-line_count:])


def write_jsonl(path: Path, rows: Iterable[CompletedExperiment]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            payload = asdict(row)
            handle.write(json.dumps(payload, sort_keys=True))
            handle.write("\n")


def write_markdown_summary(
    path: Path,
    *,
    baseline: CompletedExperiment,
    sweep_results: list[CompletedExperiment],
    canonical_results: list[CompletedExperiment],
    execution_tree: Path,
    worktree_path: Path,
    fell_back_to_current_tree: bool,
) -> None:
    successful = [row for row in sweep_results if row.success and row.result]
    ranked = rank_results([(row.name, row.result, row.success) for row in successful])
    lines = [
        "# Autoresearch Harness Summary",
        "",
        f"- Timestamp: {datetime.now(timezone.utc).isoformat()}",
        f"- Execution tree: `{execution_tree}`",
        f"- Worktree: `{worktree_path}`",
        f"- Fell back to current tree: `{fell_back_to_current_tree}`",
        f"- Sweep experiments: {len(sweep_results)}",
        f"- Successful sweep experiments: {len(successful)}",
        "",
        "## Baseline",
        "",
    ]

    if baseline.result:
        lines.append(
            f"- `baseline`: {baseline.result.decode_median:.1f} tok/s median, "
            f"{baseline.result.ttft_median_ms:.2f} ms TTFT, success={baseline.success}"
        )
    else:
        lines.append(f"- `baseline`: failed ({baseline.error})")

    lines.extend(["", "## Top Sweep Results", ""])
    if ranked:
        for name, result, _success in ranked[:10]:
            lines.append(
                f"- `{name}`: {result.decode_median:.1f} tok/s median, "
                f"{result.ttft_median_ms:.2f} ms TTFT"
            )
    else:
        lines.append("- No successful sweep results.")

    lines.extend(["", "## 5-Run Rechecks", ""])
    if canonical_results:
        for row in canonical_results:
            if row.result:
                lines.append(
                    f"- `{row.name}`: {row.result.decode_median:.1f} tok/s median, "
                    f"is_canonical_run={row.result.is_canonical_run}, success={row.success}"
                )
            else:
                lines.append(f"- `{row.name}`: failed ({row.error})")
    else:
        lines.append("- No canonical rechecks were run.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a bounded EdgeRunner autoresearch sweep against the publishable benchmark."
    )
    parser.add_argument("--count", type=int, default=100, help="number of non-canonical sweep experiments")
    parser.add_argument(
        "--canonical-top-k",
        type=int,
        default=5,
        help="re-run this many top sweep candidates with canonical publishable settings",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help="optional output directory. Defaults to benchmarks/autoresearch_runs/<timestamp>",
    )
    parser.add_argument(
        "--keep-worktree",
        action="store_true",
        help="keep the disposable worktree on disk after completion",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    contract_path = repo_root / "benchmarks" / "pinned_qwen3_0.6b_q8_0.json"
    contract = load_contract(contract_path)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifacts_dir = (
        args.artifacts_dir.resolve()
        if args.artifacts_dir
        else repo_root / "benchmarks" / "autoresearch_runs" / timestamp
    )
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    worktree_path, cleanup = create_worktree(repo_root, keep=args.keep_worktree)
    execution_tree = worktree_path
    fell_back_to_current_tree = False
    sweep_results: list[CompletedExperiment] = []
    canonical_results: list[CompletedExperiment] = []

    try:
        build_stdout = artifacts_dir / "build.stdout.log"
        build_stderr = artifacts_dir / "build.stderr.log"
        build_exit, _build_duration = run_command(
            ["swift", "build", "-c", "release"],
            cwd=execution_tree,
            env=os.environ.copy(),
            stdout_path=build_stdout,
            stderr_path=build_stderr,
        )
        if build_exit != 0:
            fell_back_to_current_tree = True
            execution_tree = repo_root
            print(
                "worktree HEAD build failed; falling back to current checkout for execution",
                file=sys.stderr,
            )
            fallback_stdout = artifacts_dir / "build.current.stdout.log"
            fallback_stderr = artifacts_dir / "build.current.stderr.log"
            build_exit, _build_duration = run_command(
                ["swift", "build", "-c", "release"],
                cwd=execution_tree,
                env=os.environ.copy(),
                stdout_path=fallback_stdout,
                stderr_path=fallback_stderr,
            )
            if build_exit != 0:
                print(tail_text(fallback_stderr) or tail_text(fallback_stdout), file=sys.stderr)
                return build_exit

        baseline = run_single_experiment(
            repo_root=repo_root,
            worktree_path=execution_tree,
            artifacts_dir=artifacts_dir,
            contract=contract,
            experiment=Experiment(name="baseline_canonical", env={}),
            run_count=contract.run_count,
            context_window=contract.context_window,
            token_count=contract.token_count,
        )

        sweep_experiments = generate_experiments(limit=args.count)
        for index, experiment in enumerate(sweep_experiments, start=1):
            row = run_single_experiment(
                repo_root=repo_root,
                worktree_path=execution_tree,
                artifacts_dir=artifacts_dir,
                contract=contract,
                experiment=experiment,
                run_count=1,
                context_window=contract.context_window,
                token_count=contract.token_count,
            )
            sweep_results.append(row)
            status = "PASS" if row.success else "FAIL"
            throughput = f"{row.result.decode_median:.1f}" if row.result else "n/a"
            print(f"[{index}/{len(sweep_experiments)}] {status} {experiment.name} median={throughput} tok/s")

        successful = [row for row in sweep_results if row.success and row.result]
        ranked = rank_results([(row.name, row.result, row.success) for row in successful])
        top_names = [name for name, _result, _success in ranked[: args.canonical_top_k]]
        top_lookup = {row.name: row for row in successful}
        for name in top_names:
            row = top_lookup[name]
            canonical_row = run_single_experiment(
                repo_root=repo_root,
                worktree_path=execution_tree,
                artifacts_dir=artifacts_dir,
                contract=contract,
                experiment=Experiment(name=f"{name}__canonical", env=row.env),
                run_count=contract.run_count,
                context_window=contract.context_window,
                token_count=contract.token_count,
            )
            canonical_results.append(canonical_row)
            throughput = f"{canonical_row.result.decode_median:.1f}" if canonical_row.result else "n/a"
            print(f"[canonical] {'PASS' if canonical_row.success else 'FAIL'} {name} median={throughput} tok/s")

        write_jsonl(artifacts_dir / "sweep_results.jsonl", sweep_results)
        write_jsonl(artifacts_dir / "canonical_results.jsonl", canonical_results)
        write_markdown_summary(
            artifacts_dir / "SUMMARY.md",
            baseline=baseline,
            sweep_results=sweep_results,
            canonical_results=canonical_results,
            execution_tree=execution_tree,
            worktree_path=worktree_path,
            fell_back_to_current_tree=fell_back_to_current_tree,
        )
        return 0
    finally:
        cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
