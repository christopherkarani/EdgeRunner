#!/bin/bash
# EdgeRunner Autoresearch Loop (Throughput Optimized)

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

ITERATIONS=${1:-100}
LOG_FILE="benchmarks/logs/results.json"

echo "=== EdgeRunner Autoresearch Loop (Throughput) ==="
echo "Target: Maximize decode throughput. Running $ITERATIONS iterations."

# Step: Ensure logs
mkdir -p benchmarks/logs

for i in $(seq 1 "$ITERATIONS"); do
    echo "Iteration $i/$ITERATIONS"

    # Run benchmark and capture output
    BENCH_OUT=$(swift test -c release --filter "PublishableBenchmark/fullBenchmark" 2>&1)

    MEDIAN=$(echo "$BENCH_OUT" | grep "qwen_decode_throughput_median" | awk '{print $3}')
    MAX=$(echo "$BENCH_OUT" | grep "qwen_decode_throughput_max" | awk '{print $3}')
    TTFT=$(echo "$BENCH_OUT" | grep "qwen_ttft_median" | awk '{print $3}')

    # Save to JSON
    TIMESTAMP=$(date +%s)
    cat <<EOF >> "$LOG_FILE"
{"timestamp": $TIMESTAMP, "iteration": $i, "median_toks": $MEDIAN, "max_toks": $MAX, "ttft": $TTFT}
EOF

    echo "  Result: $MEDIAN tok/s"
done

