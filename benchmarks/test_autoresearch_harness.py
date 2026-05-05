import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from autoresearch_harness import (
    BenchmarkResult,
    generate_experiments,
    load_benchmark_result,
    rank_results,
)


class AutoresearchHarnessTests(unittest.TestCase):
    def test_generate_experiments_puts_baseline_first(self) -> None:
        experiments = generate_experiments(limit=12)

        self.assertEqual(len(experiments), 12)
        self.assertEqual(experiments[0].name, "baseline")
        self.assertEqual(experiments[0].env, {})

    def test_generate_experiments_produces_unique_names(self) -> None:
        experiments = generate_experiments(limit=100)

        self.assertEqual(len(experiments), 100)
        self.assertEqual(len({experiment.name for experiment in experiments}), 100)

    def test_load_benchmark_result_reads_publishable_json(self) -> None:
        payload = {
            "decode_throughput": {
                "median": 248.2,
                "max": 254.2,
                "mean": 247.1,
                "stddev": 2.8,
                "min": 240.8,
            },
            "ttft_ms": {
                "median": 3.4,
                "mean": 3.5,
            },
            "memory_mb": {
                "model_load": 265.0,
                "peak_rss": 271.0,
            },
            "deterministic": True,
            "greedy_prefix": [1, 1479, 35],
            "token_hash": "0afae14a84cf0df8",
            "is_canonical_run": False,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "report.json"
            report_path.write_text(json.dumps(payload), encoding="utf-8")

            result = load_benchmark_result(report_path)

        self.assertEqual(result.decode_median, 248.2)
        self.assertEqual(result.decode_max, 254.2)
        self.assertEqual(result.ttft_median_ms, 3.4)
        self.assertEqual(result.peak_rss_mb, 271.0)
        self.assertTrue(result.deterministic)
        self.assertEqual(result.greedy_prefix, [1, 1479, 35])
        self.assertEqual(result.token_hash, "0afae14a84cf0df8")
        self.assertFalse(result.is_canonical_run)

    def test_rank_results_prefers_success_then_throughput(self) -> None:
        slower_success = BenchmarkResult(
            decode_median=240.0,
            decode_max=245.0,
            decode_mean=241.0,
            decode_stddev=1.1,
            decode_min=238.0,
            ttft_median_ms=3.6,
            ttft_mean_ms=3.7,
            model_load_mb=265.0,
            peak_rss_mb=270.0,
            deterministic=True,
            greedy_prefix=[1, 1479, 35],
            token_hash="0afae14a84cf0df8",
            is_canonical_run=False,
        )
        faster_success = BenchmarkResult(
            decode_median=250.0,
            decode_max=253.0,
            decode_mean=249.0,
            decode_stddev=1.3,
            decode_min=247.0,
            ttft_median_ms=3.5,
            ttft_mean_ms=3.5,
            model_load_mb=265.0,
            peak_rss_mb=269.0,
            deterministic=True,
            greedy_prefix=[1, 1479, 35],
            token_hash="0afae14a84cf0df8",
            is_canonical_run=False,
        )
        failed = BenchmarkResult(
            decode_median=999.0,
            decode_max=1000.0,
            decode_mean=999.0,
            decode_stddev=0.1,
            decode_min=998.0,
            ttft_median_ms=1.0,
            ttft_mean_ms=1.0,
            model_load_mb=265.0,
            peak_rss_mb=269.0,
            deterministic=False,
            greedy_prefix=[1, 1479, 999],
            token_hash="wrong",
            is_canonical_run=False,
        )

        ranked = rank_results(
            [
                ("failed", failed, False),
                ("slower_success", slower_success, True),
                ("faster_success", faster_success, True),
            ]
        )

        self.assertEqual(ranked[0][0], "faster_success")
        self.assertEqual(ranked[1][0], "slower_success")
        self.assertEqual(ranked[2][0], "failed")


if __name__ == "__main__":
    unittest.main()
