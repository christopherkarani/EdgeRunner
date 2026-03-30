#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DEFAULT_CONTRACT_PATH="$REPO_DIR/benchmarks/pinned_qwen3_0.6b_q8_0.json"
CONTRACT_PATH="${EDGERUNNER_BENCHMARK_CONTRACT:-$DEFAULT_CONTRACT_PATH}"
ALLOW_DOWNLOAD="${AUTORESEARCH_ALLOW_MODEL_DOWNLOAD:-1}"
CURL_RETRIES="${AUTORESEARCH_MODEL_DOWNLOAD_RETRIES:-5}"

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        printf 'Missing required command: %s\n' "$1" >&2
        exit 1
    fi
}

require_contract() {
    if [ ! -f "$CONTRACT_PATH" ]; then
        printf 'Benchmark contract not found: %s\n' "$CONTRACT_PATH" >&2
        exit 1
    fi
}

contract_value() {
    jq -r "$1" "$CONTRACT_PATH"
}

require_command curl
require_command jq
require_command mktemp
require_command mv
require_command rm
require_command mkdir
require_command stat
require_command shasum
require_contract

CONTRACT_MODEL_PATH="$(contract_value '.model.local_path')"
MODEL_DIR="${AUTORESEARCH_MODEL_DIR:-$(dirname "$CONTRACT_MODEL_PATH")}"
MODEL_FILENAME="${AUTORESEARCH_MODEL_FILENAME:-$(basename "$CONTRACT_MODEL_PATH")}"
MODEL_PATH="${AUTORESEARCH_MODEL_PATH:-$MODEL_DIR/$MODEL_FILENAME}"
MODEL_URL="${AUTORESEARCH_MODEL_URL:-$(contract_value '.model.download_url')}"
EXPECTED_SIZE_BYTES="${AUTORESEARCH_MODEL_EXPECTED_SIZE_BYTES:-$(contract_value '.model.size_bytes')}"
EXPECTED_SHA256="${AUTORESEARCH_MODEL_EXPECTED_SHA256:-$(contract_value '.model.sha256')}"

file_size_bytes() {
    stat -f '%z' "$1"
}

file_sha256() {
    shasum -a 256 "$1" | awk '{print $1}'
}

fail_missing_model() {
    printf '%s\n' \
        "Pinned benchmark model unavailable at $MODEL_PATH." \
        "Set AUTORESEARCH_ALLOW_MODEL_DOWNLOAD=1 to let the loop fetch it automatically," \
        "or place the pinned GGUF at that path before benchmarking." >&2
    exit 1
}

verify_existing_model() {
    local path="$1"
    if [ ! -f "$path" ]; then
        return 1
    fi

    local actual_size
    actual_size="$(file_size_bytes "$path")"
    if [ "$actual_size" != "$EXPECTED_SIZE_BYTES" ]; then
        printf 'Pinned benchmark model has unexpected size at %s (got %s, expected %s); refreshing.\n' \
            "$path" "$actual_size" "$EXPECTED_SIZE_BYTES" >&2
        return 1
    fi

    local actual_sha256
    actual_sha256="$(file_sha256 "$path")"
    if [ "$actual_sha256" != "$EXPECTED_SHA256" ]; then
        printf 'Pinned benchmark model has unexpected sha256 at %s (got %s, expected %s); refreshing.\n' \
            "$path" "$actual_sha256" "$EXPECTED_SHA256" >&2
        return 1
    fi

    printf 'Pinned benchmark model ready: %s (%s bytes, sha256 %s)\n' "$path" "$actual_size" "$actual_sha256"
    return 0
}

download_model() {
    local tmp_file
    tmp_file="$(mktemp "$MODEL_DIR/.${MODEL_FILENAME}.download.XXXXXX")"
    trap 'rm -f -- "$tmp_file"' EXIT

    printf 'Downloading pinned benchmark model to %s\n' "$MODEL_PATH"
    curl \
        --fail \
        --location \
        --retry "$CURL_RETRIES" \
        --retry-delay 2 \
        --retry-all-errors \
        --output "$tmp_file" \
        "$MODEL_URL"

    local downloaded_size
    downloaded_size="$(file_size_bytes "$tmp_file")"
    if [ "$downloaded_size" != "$EXPECTED_SIZE_BYTES" ]; then
        printf 'Downloaded benchmark model has unexpected size (got %s, expected %s).\n' \
            "$downloaded_size" "$EXPECTED_SIZE_BYTES" >&2
        exit 1
    fi

    local downloaded_sha256
    downloaded_sha256="$(file_sha256 "$tmp_file")"
    if [ "$downloaded_sha256" != "$EXPECTED_SHA256" ]; then
        printf 'Downloaded benchmark model has unexpected sha256 (got %s, expected %s).\n' \
            "$downloaded_sha256" "$EXPECTED_SHA256" >&2
        exit 1
    fi

    mv -f "$tmp_file" "$MODEL_PATH"
    trap - EXIT
    printf 'Pinned benchmark model installed: %s (%s bytes, sha256 %s)\n' \
        "$MODEL_PATH" "$downloaded_size" "$downloaded_sha256"
}

mkdir -p "$MODEL_DIR"

if verify_existing_model "$MODEL_PATH"; then
    exit 0
fi

if [ "$ALLOW_DOWNLOAD" != "1" ]; then
    fail_missing_model
fi

rm -f -- "$MODEL_PATH"
download_model

verify_existing_model "$MODEL_PATH" >/dev/null
