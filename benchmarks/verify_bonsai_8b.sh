#!/usr/bin/env bash
set -euo pipefail
MODEL=/Users/chriskarani/edgerunner-models/Bonsai-8B-Q1_0.gguf
OUT=benchmarks/autoresearch_runs/bonsai_8b_verify
mkdir -p "$OUT"

run() {
    local name=$1
    shift
    echo "=== $name ==="
    env "$@" \
        EDGERUNNER_BONSAI_MODEL_PATH=$MODEL \
        EDGERUNNER_BONSAI_OUTPUT_JSON=$(pwd)/$OUT/${name}.json \
        EDGERUNNER_BONSAI_RUNS=5 \
        EDGERUNNER_BONSAI_TOKENS=128 \
        swift test -c release --skip-build --filter "BonsaiBenchmark/bonsaiEndToEndBenchmark" 2>&1 | grep -E "Run [0-9]|Median|BONSAI_MEDIAN"
}

# Run baseline twice (cold, warm) then the top candidate
run baseline_1
run baseline_2
run top_no_kv_barrier__metal4_decode \
    EDGERUNNER_DECODE_DISABLE_KV_BARRIER=1 \
    EDGERUNNER_DECODE_PREFER_METAL4=1
run top_no_fused_final__metal4_prefill \
    EDGERUNNER_DECODE_DISABLE_FUSED_FINAL_NORM_LM_HEAD=1 \
    EDGERUNNER_PREFILL_PREFER_METAL4=1
run baseline_3

echo ""
echo "=== Summary ==="
for f in $OUT/*.json; do
    name=$(basename "$f" .json)
    median=$(python3 -c "import json; p=json.load(open('$f')); print(f\"{p['decode_throughput']['median']:.2f}\")")
    echo "$name: $median tok/s"
done
