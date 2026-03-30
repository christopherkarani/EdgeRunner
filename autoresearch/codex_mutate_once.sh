#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

ITERATION="${AUTORESEARCH_ITERATION:-${1:-0}}"
RUN_DIR="${AUTORESEARCH_RUN_DIR:-benchmarks/logs/manual}"
AUTORESEARCH_MUTATION_MODEL="${AUTORESEARCH_MUTATION_MODEL:-gpt-5.4-mini}"
AUTORESEARCH_MUTATION_REASONING_EFFORT="${AUTORESEARCH_MUTATION_REASONING_EFFORT:-xhigh}"
mkdir -p "$RUN_DIR"

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        printf 'Missing required command: %s\n' "$1" >&2
        exit 1
    fi
}

require_command codex
require_command git
require_command rsync

if [ "${AUTORESEARCH_MUTATION_DRY_RUN:-0}" = "1" ]; then
    echo "Codex mutation dry run for iteration $ITERATION"
    echo "Repo: $REPO_DIR"
    echo "Run dir: $RUN_DIR"
    echo "Model: $AUTORESEARCH_MUTATION_MODEL"
    echo "Reasoning effort: $AUTORESEARCH_MUTATION_REASONING_EFFORT"
    exit 0
fi

PROMPT_FILE="${AUTORESEARCH_MUTATION_PROMPT_FILE:-autoresearch/prompts/codex_mutation_prompt.md}"
if [ ! -f "$PROMPT_FILE" ]; then
    printf 'Missing mutation prompt file: %s\n' "$PROMPT_FILE" >&2
    exit 1
fi

SCIENTIFIC_SKILL_FILE="${AUTORESEARCH_SCIENTIFIC_SKILL_FILE:-/Users/chriskarani/.codex/skills/scientific-critical-thinking/SKILL.md}"
INFERENCE_SKILL_FILE="${AUTORESEARCH_INFERENCE_SKILL_FILE:-/Users/chriskarani/.codex/skills/inference-optimization-patterns/SKILL.md}"
if [ ! -f "$SCIENTIFIC_SKILL_FILE" ]; then
    printf 'Missing scientific thinking skill file: %s\n' "$SCIENTIFIC_SKILL_FILE" >&2
    exit 1
fi
if [ ! -f "$INFERENCE_SKILL_FILE" ]; then
    printf 'Missing inference optimization skill file: %s\n' "$INFERENCE_SKILL_FILE" >&2
    exit 1
fi

LAST_MESSAGE_FILE="$RUN_DIR/mutation-${ITERATION}-last-message.md"
SESSION_LOG="$RUN_DIR/mutation-${ITERATION}-codex.jsonl"
SESSION_ERR_LOG="$RUN_DIR/mutation-${ITERATION}-codex.stderr.log"
EXPERIMENT_QUEUE_FILE_REL="${AUTORESEARCH_EXPERIMENT_QUEUE_FILE:-autoresearch/experiment_queue.md}"
case "$EXPERIMENT_QUEUE_FILE_REL" in
    /*) EXPERIMENT_QUEUE_FILE="$EXPERIMENT_QUEUE_FILE_REL" ;;
    *) EXPERIMENT_QUEUE_FILE="$REPO_DIR/$EXPERIMENT_QUEUE_FILE_REL" ;;
esac

if [ ! -f "$EXPERIMENT_QUEUE_FILE" ]; then
    EXPERIMENT_QUEUE_FILE="$REPO_DIR/tasks/todo.md"
    echo "Warning: experiment queue file missing; falling back to task plan at $EXPERIMENT_QUEUE_FILE" >&2
fi

CODEX_HOME_SOURCE="${AUTORESEARCH_CODEX_HOME:-/Users/chriskarani/.codex}"
CODEX_HOME_WORKDIR="$(mktemp -d "$RUN_DIR/codex-home.XXXXXX")"
cleanup_codex_home() {
    rm -rf -- "$CODEX_HOME_WORKDIR"
}
trap cleanup_codex_home EXIT

mkdir -p "$CODEX_HOME_WORKDIR"
while IFS= read -r item; do
    base="$(basename "$item")"
    if [ "$base" = "skills" ]; then
        continue
    fi
    ln -s "$item" "$CODEX_HOME_WORKDIR/$base"
done < <(find "$CODEX_HOME_SOURCE" -mindepth 1 -maxdepth 1)

mkdir -p "$CODEX_HOME_WORKDIR/skills"
rsync -a --delete --exclude 'hive-expert' "$CODEX_HOME_SOURCE/skills/" "$CODEX_HOME_WORKDIR/skills/"

prompt="$(python3 - "$PROMPT_FILE" <<'PY'
import os
import pathlib
import sys

prompt_path = pathlib.Path(sys.argv[1])
text = prompt_path.read_text()
replacements = {
    "{{ITERATION}}": os.environ.get("AUTORESEARCH_ITERATION", "0"),
    "{{REPO_DIR}}": os.getcwd(),
    "{{CURRENT_HEAD}}": os.popen("git rev-parse --short HEAD").read().strip(),
    "{{BENCHMARK_JSON}}": "benchmarks/publishable_benchmark.json",
    "{{BENCHMARK_LOG}}": "benchmarks/experiment_log.md",
    "{{TODO_FILE}}": "tasks/todo.md",
    "{{EXPERIMENT_QUEUE}}": os.environ.get("AUTORESEARCH_EXPERIMENT_QUEUE_FILE", "autoresearch/experiment_queue.md"),
    "{{SCIENTIFIC_SKILL}}": os.environ.get("AUTORESEARCH_SCIENTIFIC_SKILL_FILE", "/Users/chriskarani/.codex/skills/scientific-critical-thinking/SKILL.md"),
    "{{INFERENCE_SKILL}}": os.environ.get("AUTORESEARCH_INFERENCE_SKILL_FILE", "/Users/chriskarani/.codex/skills/inference-optimization-patterns/SKILL.md"),
}
for key, value in replacements.items():
    text = text.replace(key, value)
sys.stdout.write(text)
PY
)"

set -o pipefail

printf '%s\n' "$prompt" | CODEX_HOME="$CODEX_HOME_WORKDIR" codex exec \
    --full-auto \
    --cd "$REPO_DIR" \
    --output-last-message "$LAST_MESSAGE_FILE" \
    --json \
    --model "$AUTORESEARCH_MUTATION_MODEL" \
    --config "reasoning.effort=$AUTORESEARCH_MUTATION_REASONING_EFFORT" \
    2> >(tee "$SESSION_ERR_LOG" >&2) \
    | tee "$SESSION_LOG"
codex_exit="${PIPESTATUS[0]}"

if [ "$codex_exit" -ne 0 ]; then
    printf 'Codex mutation failed for iteration %s with exit code %s\n' "$ITERATION" "$codex_exit" >&2
    exit "$codex_exit"
fi

echo "Codex mutation completed for iteration $ITERATION"
echo "Last message: $LAST_MESSAGE_FILE"
echo "Session log: $SESSION_LOG"
