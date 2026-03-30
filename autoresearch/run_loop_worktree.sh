#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

ITERATIONS="${1:-${AUTORESEARCH_MAX_EXPERIMENTS:-100}}"
WORKTREE_ROOT="${AUTORESEARCH_WORKTREE_ROOT:-${TMPDIR:-/tmp}/edgerunner-autoresearch-worktrees}"
WORKTREE_BRANCH="${AUTORESEARCH_WORKTREE_BRANCH:-autoresearch/$(date -u +%Y%m%d-%H%M%S)}"
KEEP_WORKTREE="${AUTORESEARCH_KEEP_WORKTREE:-0}"
SAFE_BRANCH="${WORKTREE_BRANCH//\//-}"
OUTER_LOG_DIR="${AUTORESEARCH_OUTER_LOG_DIR:-$REPO_DIR/benchmarks/logs/worktrees/$SAFE_BRANCH}"

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        printf 'Missing required command: %s\n' "$1" >&2
        exit 1
    fi
}

require_command git
require_command rsync

if ! [[ "$ITERATIONS" =~ ^[0-9]+$ ]] || [ "$ITERATIONS" -lt 1 ]; then
    printf 'Iteration count must be a positive integer, got: %s\n' "$ITERATIONS" >&2
    exit 1
fi

SOURCE_COMMIT="$(git rev-parse HEAD)"
SOURCE_BRANCH="$(git symbolic-ref --quiet --short HEAD 2>/dev/null || echo "detached")"
SOURCE_STATUS="$(git status --porcelain=v1)"

mkdir -p "$WORKTREE_ROOT"
WORKTREE_DIR="$(mktemp -d "$WORKTREE_ROOT/edgerunner.XXXXXX")"
export AUTORESEARCH_WORKTREE_DIR="$WORKTREE_DIR"

cleanup() {
    if [ "$KEEP_WORKTREE" = "1" ]; then
        return
    fi

    if [ -d "$WORKTREE_DIR" ]; then
        git -C "$REPO_DIR" worktree remove --force "$WORKTREE_DIR" >/dev/null 2>&1 || rm -rf -- "$WORKTREE_DIR"
    fi
}

trap cleanup EXIT

git worktree add -b "$WORKTREE_BRANCH" "$WORKTREE_DIR" "$SOURCE_COMMIT" >/dev/null
rsync -a --delete \
    --exclude '.git' \
    --exclude 'benchmarks/logs/' \
    --exclude '.DS_Store' \
    "$REPO_DIR/" "$WORKTREE_DIR/"

echo "=== EdgeRunner Autoresearch Worktree Launcher ==="
echo "Source branch: $SOURCE_BRANCH"
echo "Source commit: $SOURCE_COMMIT"
echo "Worktree branch: $WORKTREE_BRANCH"
echo "Worktree path: $WORKTREE_DIR"
echo "Log dir: $OUTER_LOG_DIR"
if [ -n "$SOURCE_STATUS" ]; then
    echo "Warning: source checkout has uncommitted changes; the worktree was seeded from the current checkout snapshot."
fi
echo
echo "Launching autoresearch loop from isolated worktree..."

status=0
if AUTORESEARCH_IN_WORKTREE=1 AUTORESEARCH_LOG_DIR="$OUTER_LOG_DIR" ./autoresearch/run_loop.sh "$ITERATIONS"; then
    status=0
else
    status=$?
fi

exit "$status"
