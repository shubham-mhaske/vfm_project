#!/bin/bash
#
# Runs the full evaluation pipeline for a given experiment run directory.
# 1. Finds the best checkpoint based on validation scores.
# 2. Finds the optimal per-class thresholds for that checkpoint.
# 3. Runs final evaluation on the test set using TTA and optimal thresholds.
#
# Usage:
#   bash scripts/evaluation/evaluate_experiment.sh <path_to_run_dir>
#
# Example:
#   bash scripts/evaluation/evaluate_experiment.sh finetune_logs/path_sam2_focal-2025-11-27_12-00-00
#

set -e

RUN_DIR=$1

if [ -z "$RUN_DIR" ]; then
    echo "Usage: $0 <path_to_run_dir>"
    exit 1
fi

if [ ! -d "$RUN_DIR" ]; then
    echo "Error: Run directory not found at '$RUN_DIR'"
    exit 1
fi

echo "=============================================="
echo "Full Evaluation Pipeline"
echo "=============================================="
echo "Run Directory: $RUN_DIR"
echo "=============================================="


# --- 1. Find Best Checkpoint ---
echo -e "\n[1/3] Finding best checkpoint..."
# The script prints the path as its last output line
BEST_CKPT=$(python scripts/validation/find_best_checkpoint.py --run_dir "$RUN_DIR" | tail -n 1)

if [ -z "$BEST_CKPT" ] || [ ! -f "$BEST_CKPT" ]; then
    echo "Error: Failed to find the best checkpoint."
    exit 1
fi


# --- 2. Find Optimal Thresholds ---
echo -e "\n[2/3] Finding optimal thresholds for a checkpoint..."
THRESH_CONFIG="configs/optimal_thresholds_${SLURM_JOB_ID:-local}.json"
python scripts/validation/find_optimal_thresholds.py \
    --checkpoint "$BEST_CKPT" \
    --split val \
    --output "$THRESH_CONFIG"

if [ ! -f "$THRESH_CONFIG" ]; then
    echo "Error: Failed to create threshold config."
    exit 1
fi
echo "Optimal thresholds saved to $THRESH_CONFIG"


# --- 3. Run Final Evaluation ---
echo -e "\n[3/3] Running final evaluation on the test set..."
EVAL_OUTPUT_DIR="results/$(basename "$RUN_DIR")_final_eval"
python src/evaluate_segmentation.py \
    --checkpoint "$BEST_CKPT" \
    --split test \
    --use_tta \
    --threshold_config "$THRESH_CONFIG" \
    --output_dir "$EVAL_OUTPUT_DIR"

echo -e "\n=============================================="
echo "✅ Evaluation Complete!"
echo "Final results saved in: $EVAL_OUTPUT_DIR"
echo "=============================================="
