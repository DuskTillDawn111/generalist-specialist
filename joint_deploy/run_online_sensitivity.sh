#!/usr/bin/env bash

set -euo pipefail

ROOT="/root/autodl-tmp/generalist-specialist"

cd "$ROOT"

POOL="benchmark/task_pools/mixed_formal_800.jsonl"
RESULT_DIR="benchmark/results/joint/sensitivity"

mkdir -p "$RESULT_DIR"


run_case() {
    local alpha="$1"
    local window="$2"
    local tag="$3"

    local output="${RESULT_DIR}/online_${tag}.jsonl"
    local summary="${RESULT_DIR}/online_${tag}.summary.json"
    local log="${RESULT_DIR}/online_${tag}.log"

    echo
    echo "========================================"
    echo "CASE: ${tag}"
    echo "alpha=${alpha}"
    echo "window=${window}"
    echo "========================================"

    if [ -f "$summary" ]; then
        echo "Already finished:"
        echo "$summary"
        return
    fi

    python joint_deploy/run_online_adaptive.py \
        --pool "$POOL" \
        --output "$output" \
        --alpha "$alpha" \
        --max-outstanding "$window" \
        --timeout-s 3600 \
        2>&1 | tee "$log"
}


# alpha sensitivity, window fixed at 32
run_case 0.10 32 "a010_w32"

# window sensitivity, alpha fixed at 0.25
run_case 0.25 8  "a025_w08"
run_case 0.25 16 "a025_w16"

# a025_w32 already exists from the main experiment,
# so we do not repeat it here.

run_case 0.25 64 "a025_w64"

# alpha sensitivity, window fixed at 32
run_case 0.50 32 "a050_w32"


echo
echo "========================================"
echo "Sensitivity sweep finished."
echo "========================================"
