#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

ITERATIONS="${1:-${AUTORESEARCH_MAX_EXPERIMENTS:-100}}"
BENCHMARK_COMMAND="${AUTORESEARCH_BENCHMARK_COMMAND:-swift test -c release --filter \"PublishableBenchmark/fullBenchmark\"}"
EXPERIMENT_COMMAND="${AUTORESEARCH_EXPERIMENT_COMMAND-./autoresearch/codex_mutate_once.sh}"
MODEL_BOOTSTRAP_COMMAND="${AUTORESEARCH_MODEL_BOOTSTRAP_COMMAND-./autoresearch/ensure_benchmark_model.sh}"
RESULT_JSON="${AUTORESEARCH_RESULT_JSON:-benchmarks/publishable_benchmark.json}"
CONTRACT_PATH="${EDGERUNNER_BENCHMARK_CONTRACT:-$REPO_DIR/benchmarks/pinned_qwen3_0.6b_q8_0.json}"
LOG_DIR="${AUTORESEARCH_LOG_DIR:-benchmarks/logs}"
RUN_DIR="$LOG_DIR/runs"
SUMMARY_LOG="${AUTORESEARCH_SUMMARY_LOG:-$LOG_DIR/results.jsonl}"
BEST_REPORT="${AUTORESEARCH_BEST_REPORT:-$LOG_DIR/best-run.json}"
BASELINE_HEALTH_ENABLED="${AUTORESEARCH_ENFORCE_BASELINE_HEALTH:-1}"
BASELINE_HEALTH_RUNS="${AUTORESEARCH_BASELINE_HEALTH_RUNS:-3}"
BASELINE_HEALTH_REPORT="${AUTORESEARCH_BASELINE_HEALTH_REPORT:-$LOG_DIR/baseline-health.json}"

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        printf 'Missing required command: %s\n' "$1" >&2
        exit 1
    fi
}

require_command git
require_command jq
require_command swift

if ! [[ "$ITERATIONS" =~ ^[0-9]+$ ]] || [ "$ITERATIONS" -lt 1 ]; then
    printf 'Iteration count must be a positive integer, got: %s\n' "$ITERATIONS" >&2
    exit 1
fi

if ! [[ "$BASELINE_HEALTH_RUNS" =~ ^[0-9]+$ ]] || [ "$BASELINE_HEALTH_RUNS" -lt 1 ]; then
    printf 'Baseline health run count must be a positive integer, got: %s\n' "$BASELINE_HEALTH_RUNS" >&2
    exit 1
fi

if [ ! -f "$CONTRACT_PATH" ]; then
    printf 'Benchmark contract not found: %s\n' "$CONTRACT_PATH" >&2
    exit 1
fi

TARGET_MODEL="$(jq -r '.model.name' "$CONTRACT_PATH")"
EXPECTED_TOKEN_HASH="$(jq -r '.publishable.expected_token_hash' "$CONTRACT_PATH")"
EXPECTED_GREEDY_PREFIX="$(jq -c '.publishable.expected_greedy_prefix' "$CONTRACT_PATH")"

mkdir -p "$LOG_DIR" "$RUN_DIR"

echo "=== EdgeRunner Autoresearch Loop ==="
echo "Target model: $TARGET_MODEL"
echo "Iterations: $ITERATIONS"
echo "Benchmark command: $BENCHMARK_COMMAND"
echo "Benchmark contract: $CONTRACT_PATH"
if [ "$BASELINE_HEALTH_ENABLED" = "1" ]; then
    echo "Baseline health gate: ${BASELINE_HEALTH_RUNS} canonical runs"
else
    echo "Baseline health gate: disabled"
fi
if [ -n "$EXPERIMENT_COMMAND" ]; then
    echo "Experiment command: $EXPERIMENT_COMMAND"
fi
if [ -n "$MODEL_BOOTSTRAP_COMMAND" ]; then
    echo "Model bootstrap: $MODEL_BOOTSTRAP_COMMAND"
fi
echo "Result JSON: $RESULT_JSON"
echo

if [ -n "$MODEL_BOOTSTRAP_COMMAND" ]; then
    echo "Ensuring pinned benchmark model..."
    bash -lc "$MODEL_BOOTSTRAP_COMMAND"
    echo
fi

validate_benchmark_report() {
    local report_file="$1"
    jq -e '
        .decode_throughput.median != null
        and .decode_throughput.max != null
        and .ttft_ms.median != null
        and .model != null
        and .token_hash != null
        and .greedy_prefix != null
        and .deterministic != null
        and .is_canonical_run != null
        and .runs != null
    ' "$report_file" >/dev/null 2>&1
}

logs_contain() {
    local pattern="$1"
    shift
    grep -Fqs -- "$pattern" "$@" 2>/dev/null
}

logs_match() {
    local pattern="$1"
    shift
    grep -Eqs -- "$pattern" "$@" 2>/dev/null
}

classify_benchmark_failure() {
    local stdout_file="$1"
    local stderr_file="$2"
    local failure_reason="${3:-}"

    if logs_contain "Pinned publishable benchmark model not found" "$stdout_file" "$stderr_file" \
        || logs_contain "unexpected size" "$stdout_file" "$stderr_file" \
        || logs_contain "unexpected sha256" "$stdout_file" "$stderr_file"; then
        printf '%s\n' "benchmark_model_contract_mismatch"
    elif logs_contain "Non-deterministic output across runs" "$stdout_file" "$stderr_file"; then
        printf '%s\n' "benchmark_nondeterministic"
    elif logs_contain "Benchmark input or decode path drifted" "$stdout_file" "$stderr_file"; then
        printf '%s\n' "benchmark_input_drift"
    elif logs_contain "Publishable output drifted" "$stdout_file" "$stderr_file"; then
        printf '%s\n' "benchmark_token_hash_drift"
    elif logs_contain "NaN/Inf" "$stdout_file" "$stderr_file"; then
        printf '%s\n' "benchmark_non_finite_logits"
    elif logs_contain "Context window must be >= token count" "$stdout_file" "$stderr_file"; then
        printf '%s\n' "benchmark_invalid_configuration"
    elif logs_match '(^|[[:space:]])error:' "$stdout_file" "$stderr_file"; then
        printf '%s\n' "infra_failure"
    elif [ "$failure_reason" = "missing benchmark report: $RESULT_JSON" ]; then
        printf '%s\n' "benchmark_missing_report"
    elif [ "$failure_reason" = "benchmark JSON missing required fields" ]; then
        printf '%s\n' "benchmark_json_missing_fields"
    else
        printf '%s\n' "infra_failure"
    fi
}

classify_benchmark_report() {
    local report_file="$1"

    if ! jq -e '.deterministic == true' "$report_file" >/dev/null 2>&1; then
        printf '%s\n' "benchmark_nondeterministic"
    elif ! jq -e '.is_canonical_run == true' "$report_file" >/dev/null 2>&1; then
        printf '%s\n' "benchmark_noncanonical_result"
    elif ! jq -e --argjson expected_prefix "$EXPECTED_GREEDY_PREFIX" '.greedy_prefix == $expected_prefix' "$report_file" >/dev/null 2>&1; then
        printf '%s\n' "benchmark_input_drift"
    elif ! jq -e --arg expected_hash "$EXPECTED_TOKEN_HASH" '.token_hash == $expected_hash' "$report_file" >/dev/null 2>&1; then
        printf '%s\n' "benchmark_token_hash_drift"
    elif ! jq -e '([.runs[]?.has_nan] | all(. == false))' "$report_file" >/dev/null 2>&1; then
        printf '%s\n' "benchmark_non_finite_logits"
    elif ! jq -e '.decode_throughput.median > 0' "$report_file" >/dev/null 2>&1; then
        printf '%s\n' "benchmark_invalid_metrics"
    else
        printf '%s\n' "benchmark_contract_violation"
    fi
}

validate_canonical_benchmark_report() {
    local report_file="$1"
    jq -e \
        --arg expected_hash "$EXPECTED_TOKEN_HASH" \
        --argjson expected_prefix "$EXPECTED_GREEDY_PREFIX" \
        '
        .deterministic == true
        and .is_canonical_run == true
        and .token_hash == $expected_hash
        and .greedy_prefix == $expected_prefix
        and .decode_throughput.median > 0
        and ([.runs[]?.has_nan] | all(. == false))
        ' "$report_file" >/dev/null 2>&1
}

append_run_summary() {
    local iteration="$1"
    local timestamp="$2"
    local commit_sha="$3"
    local current_report="$4"
    local output_file="$5"

    jq -c \
        --arg iteration "$iteration" \
        --arg timestamp "$timestamp" \
        --arg commit_sha "$commit_sha" \
        --arg benchmark_command "$BENCHMARK_COMMAND" \
        --arg experiment_command "$EXPERIMENT_COMMAND" \
        '
        {
            iteration: ($iteration | tonumber),
            timestamp: $timestamp,
            commit: $commit_sha,
            benchmark_command: $benchmark_command,
            experiment_command: (if $experiment_command == "" then null else $experiment_command end),
            model: .model,
            model_path: .model_path,
            model_file_size_bytes: .model_file_size_bytes,
            token_hash: .token_hash,
            deterministic: .deterministic,
            is_canonical_run: .is_canonical_run,
            greedy_prefix: .greedy_prefix,
            decode_throughput: .decode_throughput,
            ttft_ms: .ttft_ms,
            memory_mb: .memory_mb,
            tokens_per_run: .tokens_per_run,
            num_runs: .num_runs
        }
        ' "$current_report" > "$output_file"
}

update_best_report() {
    local current_report="$1"
    local current_median="$2"

    if [ ! -f "$BEST_REPORT" ]; then
        cp "$current_report" "$BEST_REPORT"
        return
    fi

    local best_median
    best_median="$(jq -r '.decode_throughput.median' "$BEST_REPORT")"

    if awk -v current="$current_median" -v best="$best_median" 'BEGIN { exit !(current > best) }'; then
        cp "$current_report" "$BEST_REPORT"
    fi
}

capture_changed_paths() {
    local output_file="$1"
    git status --porcelain=v1 \
        | awk '
            length($0) > 3 {
                path = substr($0, 4)
                if (path !~ /^benchmarks\/logs\// && path !~ /^\.DS_Store$/ && path !~ /^Sources\/\.DS_Store$/ && path !~ /^Tests\/\.DS_Store$/ && path !~ /^docs\/\.DS_Store$/) {
                    print path
                }
            }
        ' \
        | sort -u > "$output_file"
}

revert_changed_paths() {
    local pre_paths_file="$1"
    local post_paths_file="$2"
    local paths_to_revert
    paths_to_revert="$(mktemp)"

    comm -13 "$pre_paths_file" "$post_paths_file" > "$paths_to_revert" || true

    while IFS= read -r path; do
        [ -z "$path" ] && continue
        if git ls-files --error-unmatch -- "$path" >/dev/null 2>&1; then
            git restore --source=HEAD --worktree --staged -- "$path" || true
        else
            rm -rf -- "$path" || true
        fi
    done < "$paths_to_revert"

    rm -f "$paths_to_revert"
}

write_iteration_summary() {
    local summary_json="$1"
    local iteration="$2"
    local timestamp="$3"
    local commit_before="$4"
    local commit_after="$5"
    local status="$6"
    local decision="$7"
    local error_message="$8"
    local error_class="${9}"
    local report_file="${10}"

    if [ -f "$report_file" ] && [ "$status" = "success" ]; then
        jq -c \
            --arg iteration "$iteration" \
            --arg timestamp "$timestamp" \
            --arg commit_before "$commit_before" \
            --arg commit_after "$commit_after" \
            --arg status "$status" \
            --arg decision "$decision" \
            --arg error_message "$error_message" \
            --arg error_class "$error_class" \
            '
            . + {
                iteration: ($iteration | tonumber),
                timestamp: $timestamp,
                commit_before: $commit_before,
                commit_after: $commit_after,
                iteration_status: $status,
                decision: $decision,
                error: (if $error_message == "" then null else $error_message end),
                error_class: (if $error_class == "" then null else $error_class end)
            }
            ' "$report_file" > "$summary_json"
    else
        jq -n \
            --arg iteration "$iteration" \
            --arg timestamp "$timestamp" \
            --arg commit_before "$commit_before" \
            --arg commit_after "$commit_after" \
            --arg status "$status" \
            --arg decision "$decision" \
            --arg error_message "$error_message" \
            --arg error_class "$error_class" \
            '{
                iteration: ($iteration | tonumber),
                timestamp: $timestamp,
                commit_before: $commit_before,
                commit_after: $commit_after,
                iteration_status: $status,
                decision: $decision,
                error: (if $error_message == "" then null else $error_message end),
                error_class: (if $error_class == "" then null else $error_class end)
            }' > "$summary_json"
    fi
}

write_baseline_health_report() {
    local status="$1"
    local error_class="$2"
    local error_message="$3"
    local reference_hash="$4"
    local reference_prefix="$5"
    local best_decode_median="$6"

    jq -n \
        --arg status "$status" \
        --arg benchmark_command "$BENCHMARK_COMMAND" \
        --arg error_class "$error_class" \
        --arg error_message "$error_message" \
        --arg reference_hash "$reference_hash" \
        --arg best_decode_median "$best_decode_median" \
        --argjson baseline_runs "$BASELINE_HEALTH_RUNS" \
        --argjson reference_prefix "${reference_prefix:-null}" \
        '{
            status: $status,
            benchmark_command: $benchmark_command,
            baseline_runs: $baseline_runs,
            error_class: (if $error_class == "" then null else $error_class end),
            error: (if $error_message == "" then null else $error_message end),
            token_hash: (if $reference_hash == "" then null else $reference_hash end),
            greedy_prefix: $reference_prefix,
            best_decode_median: (if $best_decode_median == "" then null else ($best_decode_median | tonumber) end)
        }' > "$BASELINE_HEALTH_REPORT"
}

run_baseline_health_check() {
    if [ "$BASELINE_HEALTH_ENABLED" != "1" ]; then
        write_baseline_health_report "skipped" "" "" "" "null" ""
        return 0
    fi

    local health_dir="$LOG_DIR/baseline-health"
    mkdir -p "$health_dir"

    local reference_hash=""
    local reference_prefix=""
    local best_baseline_report=""
    local best_baseline_median=""

    echo "Running baseline health gate ($BASELINE_HEALTH_RUNS runs)..."

    for run in $(seq 1 "$BASELINE_HEALTH_RUNS"); do
        local stdout_file="$health_dir/run-$(printf '%02d' "$run").stdout.log"
        local stderr_file="$health_dir/run-$(printf '%02d' "$run").stderr.log"
        local report_file="$health_dir/run-$(printf '%02d' "$run").json"
        local benchmark_error=""
        local error_class=""

        rm -f "$RESULT_JSON"
        if ! bash -lc "$BENCHMARK_COMMAND" >"$stdout_file" 2>"$stderr_file"; then
            benchmark_error="baseline benchmark command failed; see $stdout_file and $stderr_file"
            error_class="$(classify_benchmark_failure "$stdout_file" "$stderr_file" "$benchmark_error")"
            write_baseline_health_report "unhealthy" "$error_class" "$benchmark_error" "$reference_hash" "${reference_prefix:-null}" "$best_baseline_median"
            echo "Baseline health gate failed ($error_class); see $stdout_file and $stderr_file" >&2
            return 1
        fi

        if [ ! -f "$RESULT_JSON" ]; then
            benchmark_error="missing benchmark report: $RESULT_JSON"
            error_class="$(classify_benchmark_failure "$stdout_file" "$stderr_file" "$benchmark_error")"
            write_baseline_health_report "unhealthy" "$error_class" "$benchmark_error" "$reference_hash" "${reference_prefix:-null}" "$best_baseline_median"
            echo "Baseline health gate failed ($error_class); missing benchmark report." >&2
            return 1
        fi

        if ! validate_benchmark_report "$RESULT_JSON"; then
            benchmark_error="benchmark JSON missing required fields"
            error_class="$(classify_benchmark_failure "$stdout_file" "$stderr_file" "$benchmark_error")"
            write_baseline_health_report "unhealthy" "$error_class" "$benchmark_error" "$reference_hash" "${reference_prefix:-null}" "$best_baseline_median"
            echo "Baseline health gate failed ($error_class); invalid benchmark report." >&2
            return 1
        fi

        if ! validate_canonical_benchmark_report "$RESULT_JSON"; then
            benchmark_error="baseline benchmark report did not satisfy the pinned publishable contract"
            error_class="$(classify_benchmark_report "$RESULT_JSON")"
            write_baseline_health_report "unhealthy" "$error_class" "$benchmark_error" "$reference_hash" "${reference_prefix:-null}" "$best_baseline_median"
            echo "Baseline health gate failed ($error_class); see $report_file." >&2
            cp "$RESULT_JSON" "$report_file"
            return 1
        fi

        cp "$RESULT_JSON" "$report_file"

        local token_hash
        token_hash="$(jq -r '.token_hash' "$RESULT_JSON")"
        local greedy_prefix
        greedy_prefix="$(jq -c '.greedy_prefix' "$RESULT_JSON")"
        local current_median
        current_median="$(jq -r '.decode_throughput.median' "$RESULT_JSON")"

        if [ -z "$reference_hash" ]; then
            reference_hash="$token_hash"
            reference_prefix="$greedy_prefix"
        fi

        if [ -z "$best_baseline_median" ] || awk -v current="$current_median" -v best="$best_baseline_median" 'BEGIN { exit !(current > best) }'; then
            best_baseline_median="$current_median"
            best_baseline_report="$report_file"
        fi
    done

    if [ -n "$best_baseline_report" ]; then
        cp "$best_baseline_report" "$BEST_REPORT"
    fi

    write_baseline_health_report "healthy" "" "" "$reference_hash" "$reference_prefix" "$best_baseline_median"
    echo "Baseline health gate passed."
    if [ -n "$best_baseline_median" ]; then
        printf 'Baseline best decode median: %s tok/s\n' "$best_baseline_median"
    fi
    echo
}

run_baseline_health_check

for i in $(seq 1 "$ITERATIONS"); do
    iteration_dir="$RUN_DIR/iteration-$(printf '%03d' "$i")"
    mkdir -p "$iteration_dir"

    timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    commit_before="$(git rev-parse --short HEAD)"
    stdout_file="$iteration_dir/benchmark.stdout.log"
    stderr_file="$iteration_dir/benchmark.stderr.log"
    iteration_report="$iteration_dir/publishable_benchmark.json"
    summary_json="$iteration_dir/summary.json"
    pre_paths_file="$iteration_dir/pre-change-paths.txt"
    post_paths_file="$iteration_dir/post-change-paths.txt"

    capture_changed_paths "$pre_paths_file"

    echo "Iteration $i/$ITERATIONS [$commit_before]"

    mutation_ok=true
    if [ -n "$EXPERIMENT_COMMAND" ]; then
        experiment_stdout="$iteration_dir/experiment.stdout.log"
        experiment_stderr="$iteration_dir/experiment.stderr.log"
        echo "  Running experiment hook"
        if ! AUTORESEARCH_ITERATION="$i" AUTORESEARCH_RUN_DIR="$iteration_dir" bash -lc "$EXPERIMENT_COMMAND" >"$experiment_stdout" 2>"$experiment_stderr"; then
            echo "  Experiment hook failed; see $experiment_stdout and $experiment_stderr" >&2
            mutation_ok=false
        fi
    fi

    benchmark_ok=true
    benchmark_error=""
    benchmark_error_class=""
    if [ "$mutation_ok" = true ]; then
        echo "  Running benchmark"
        rm -f "$RESULT_JSON"
        if ! bash -lc "$BENCHMARK_COMMAND" >"$stdout_file" 2>"$stderr_file"; then
            benchmark_ok=false
            benchmark_error="benchmark command failed; see $stdout_file and $stderr_file"
            benchmark_error_class="$(classify_benchmark_failure "$stdout_file" "$stderr_file" "$benchmark_error")"
            echo "  Benchmark failed; see $stdout_file and $stderr_file" >&2
        fi
    fi

    if [ "$benchmark_ok" = true ] && [ ! -f "$RESULT_JSON" ]; then
        benchmark_ok=false
        benchmark_error="missing benchmark report: $RESULT_JSON"
        benchmark_error_class="$(classify_benchmark_failure "$stdout_file" "$stderr_file" "$benchmark_error")"
        echo "  Missing benchmark report: $RESULT_JSON" >&2
    fi

    if [ "$benchmark_ok" = true ]; then
        if ! validate_benchmark_report "$RESULT_JSON"; then
            benchmark_ok=false
            benchmark_error="benchmark JSON missing required fields"
            benchmark_error_class="$(classify_benchmark_failure "$stdout_file" "$stderr_file" "$benchmark_error")"
            echo "  Benchmark JSON missing required fields; see $RESULT_JSON" >&2
        elif ! validate_canonical_benchmark_report "$RESULT_JSON"; then
            benchmark_ok=false
            benchmark_error="benchmark report did not satisfy the pinned publishable contract"
            benchmark_error_class="$(classify_benchmark_report "$RESULT_JSON")"
            echo "  Benchmark report failed pinned-contract validation; see $RESULT_JSON" >&2
        fi
    fi

    commit_after="$commit_before"
    decision="continue"
    if [ "$benchmark_ok" = true ]; then
        cp "$RESULT_JSON" "$iteration_report"
        current_median="$(jq -r '.decode_throughput.median' "$RESULT_JSON")"
        current_ttft="$(jq -r '.ttft_ms.median' "$RESULT_JSON")"
        current_hash="$(jq -r '.token_hash' "$RESULT_JSON")"
        current_prefix="$(jq -c '.greedy_prefix' "$RESULT_JSON")"

        is_best_run=false
        if [ ! -f "$BEST_REPORT" ]; then
            is_best_run=true
        else
            best_median="$(jq -r '.decode_throughput.median' "$BEST_REPORT")"
            if awk -v current="$current_median" -v best="$best_median" 'BEGIN { exit !(current > best) }'; then
                is_best_run=true
            fi
        fi

        if [ "$is_best_run" = true ]; then
            update_best_report "$iteration_report" "$current_median"
            decision="keep"
            commit_after="$(git rev-parse --short HEAD)"
            printf '  decode median: %s tok/s, ttft: %s ms, prefix: %s, hash: %s [best]\n' \
                "$current_median" "$current_ttft" "$current_prefix" "$current_hash"
        else
            decision="revert"
            echo "  Regression or non-improvement; reverting iteration changes."
        fi

        commit_after="$(git rev-parse --short HEAD)"
        if [ "$decision" = "revert" ]; then
            capture_changed_paths "$post_paths_file"
            revert_changed_paths "$pre_paths_file" "$post_paths_file"
            commit_after="$(git rev-parse --short HEAD)"
        fi
        write_iteration_summary "$summary_json" "$i" "$timestamp" "$commit_before" "$commit_after" "success" "$decision" "" "" "$iteration_report"
    else
        decision="revert"
        capture_changed_paths "$post_paths_file"
        revert_changed_paths "$pre_paths_file" "$post_paths_file"
        commit_after="$(git rev-parse --short HEAD)"
        if [ "$mutation_ok" = false ] && [ -z "$benchmark_error" ]; then
            benchmark_error="experiment hook failed"
            benchmark_error_class="mutation_failure"
        fi
        write_iteration_summary "$summary_json" "$i" "$timestamp" "$commit_before" "$commit_after" "failed" "$decision" "$benchmark_error" "$benchmark_error_class" "$iteration_report"
        echo "  Iteration failed; continuing to next experiment."
    fi

    cat "$summary_json" >> "$SUMMARY_LOG"

done

echo
echo "Best report: $BEST_REPORT"
echo "Run summaries: $SUMMARY_LOG"
