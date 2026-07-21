#!/usr/bin/env bash
# Find the most recent RSL-RL checkpoint under logs/ and play it.
#
# Usage:
#   ./scripts/rsl_rl/play_latest.sh                 # default: 20 envs, resets disabled
#   ./scripts/rsl_rl/play_latest.sh --num_envs 1    # forward any extra args to play.py

set -euo pipefail

TASK="${TASK:-MosOne-Mos20262ClosedUsd-ClosedUsd-v0}"
LOGS_DIR="${LOGS_DIR:-logs}"

# Resolve repo root so the script works from any cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [ ! -d "${LOGS_DIR}" ]; then
    echo "[play_latest] No '${LOGS_DIR}/' directory found in ${REPO_ROOT}. Train a policy first." >&2
    exit 1
fi

# Newest .pt by modification time (handles spaces / weird names safely).
# NOTE: 'head -n 1' here would SIGPIPE 'find' under pipefail once logs/ grows large;
# 'sed -n 1p' consumes the whole stream so the pipeline always exits 0.
CHECKPOINT="$(find "${LOGS_DIR}" -type f -name '*.pt' -printf '%T@ %p\n' \
              | sort -nr | sed -n '1p' | cut -d' ' -f2-)"

if [ -z "${CHECKPOINT}" ]; then
    echo "[play_latest] No .pt checkpoints found under ${LOGS_DIR}/." >&2
    exit 1
fi

echo "[play_latest] task       = ${TASK}"
echo "[play_latest] checkpoint = ${CHECKPOINT}"

# Default play args; override / extend by passing extra flags on the CLI.
if [ "$#" -eq 0 ]; then
    set -- --num_envs 20 --disable_resets
fi

exec python scripts/rsl_rl/play.py \
    --task "${TASK}" \
    --checkpoint "${CHECKPOINT}" \
    "$@"
