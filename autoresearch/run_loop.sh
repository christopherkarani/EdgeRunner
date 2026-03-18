#!/bin/bash
# EdgeRunner Autoresearch Loop
# Continuously benchmarks and tracks performance against MLX target

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

MLX_TARGET=277.8  # MLX tok/s median on 128-token decode
LLAMACPP_REF=200.3

echo "=== EdgeRunner Autoresearch Loop ==="
echo "Target: Beat MLX ($MLX_TARGET tok/s) on 128-token decode"
echo "Device: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'Apple Silicon')"
echo ""

# Step 1: Build release
echo "[BUILD] Building release..."
swift build -c release 2>&1 | tail -3

# Step 2: Run correctness test (4-token)
echo ""
echo "[CORRECTNESS] Running 4-token benchmark..."
CORRECTNESS=$(swift test -c release --filter "QwenBenchmark/decodeBenchmark" 2>&1)
if echo "$CORRECTNESS" | grep -q "passed"; then
    CORRECT_TOKS=$(echo "$CORRECTNESS" | grep "qwen_decode_throughput" | awk '{print $3}')
    echo "  PASS: $CORRECT_TOKS tok/s (4-token)"
else
    echo "  FAIL: Correctness check failed!"
    echo "$CORRECTNESS" | grep -E "Expectation|generated_tokens"
    exit 1
fi

# Step 3: Run publishable benchmark (128-token)
echo ""
echo "[BENCHMARK] Running 128-token publishable benchmark..."
BENCH=$(swift test -c release --filter "PublishableBenchmark/fullBenchmark" 2>&1)

MEDIAN=$(echo "$BENCH" | grep "qwen_decode_throughput_median" | awk '{print $3}')
MAX=$(echo "$BENCH" | grep "qwen_decode_throughput_max" | awk '{print $3}')
TTFT=$(echo "$BENCH" | grep "qwen_ttft_median" | awk '{print $3}')

echo ""
echo "=========================================="
echo "  RESULTS"
echo "=========================================="
echo "  128-token decode (median): $MEDIAN tok/s"
echo "  128-token decode (max):    $MAX tok/s"
echo "  TTFT (median):             $TTFT ms"
echo ""
echo "  vs MLX ($MLX_TARGET tok/s):  $(echo "scale=1; ($MEDIAN - $MLX_TARGET) / $MLX_TARGET * 100" | bc)%"
echo "  vs llama.cpp ($LLAMACPP_REF tok/s): $(echo "scale=1; ($MEDIAN - $LLAMACPP_REF) / $LLAMACPP_REF * 100" | bc)%"
echo "=========================================="

# Step 4: Determine if we beat targets
if (( $(echo "$MEDIAN > $MLX_TARGET" | bc -l) )); then
    echo ""
    echo "*** MILESTONE: EdgeRunner BEATS MLX! ***"
fi

if (( $(echo "$MEDIAN > $LLAMACPP_REF" | bc -l) )); then
    echo "  EdgeRunner beats llama.cpp by $(echo "scale=1; ($MEDIAN - $LLAMACPP_REF) / $LLAMACPP_REF * 100" | bc)%"
fi
