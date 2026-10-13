#!/bin/bash
set -e

POOL="benchmark/task_pools/taskmix_heavy/ocr_heavy_800.jsonl"
PLAN_DIR="benchmark/results/composition_motivation/plans"
OUT_DIR="benchmark/results/composition_motivation/runs"

mkdir -p "$OUT_DIR"

run_one() {
    NAME="$1"
    MAX_MODELS="$2"

    echo
    echo "============================================================"
    echo "RUNNING: $NAME"
    echo "MAX_MODELS: $MAX_MODELS"
    echo "============================================================"

    python joint_deploy/run_placement_plan.py \
        --placement "${PLAN_DIR}/${NAME}.json" \
        --budget-gib 24 \
        --max-models "$MAX_MODELS" \
        --pool "$POOL" \
        --output "${OUT_DIR}/${NAME}.jsonl" \
        --timeout-s 3600 \
        2>&1 | tee "${OUT_DIR}/${NAME}.log"

    echo
    echo "FINISHED: $NAME"
}

run_one d0_qwen 1
run_one d1_qwen_paddle 2
run_one d2_qwen_resnet_paddle 3
run_one d3_qwen_distil_resnet_paddle 4
run_one d4_all_specialists 4

echo
echo "============================================================"
echo "ALL 5 RUNS FINISHED"
echo "============================================================"
