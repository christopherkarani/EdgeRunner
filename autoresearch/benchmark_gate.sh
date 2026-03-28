#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PROMPT_TOKENS="${EDGERUNNER_GATE_PROMPT_TOKENS:-1024}"
GENERATE_TOKENS="${EDGERUNNER_GATE_GENERATE_TOKENS:-128}"
RUNS="${EDGERUNNER_GATE_RUNS:-1}"
OUTPUT_PATH="${EDGERUNNER_GATE_OUTPUT_PATH:-$ROOT_DIR/benchmarks/long_prompt_framework_comparison.dev.json}"

echo "[gate] build"
swift build -c release

echo "[gate] publishable benchmark"
swift test -c release --filter "PublishableBenchmark/fullBenchmark"

echo "[gate] long-prompt benchmark"
python3 benchmarks/run_long_prompt_framework_benchmark.py \
  --prompt-tokens "$PROMPT_TOKENS" \
  --generate-tokens "$GENERATE_TOKENS" \
  --runs "$RUNS" \
  --output "$OUTPUT_PATH"

if [[ "${EDGERUNNER_GATE_RUN_PARITY:-0}" == "1" ]]; then
  echo "[gate] parity"
  swift test -c release --filter "QwenLatePrefixParityTest"
fi

echo "[gate] complete"
