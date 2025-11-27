#!/bin/bash
#
# Finds the optimal confidence threshold for each class on the validation set
# and saves the result to a JSON config file.
#
# Usage:
#   bash scripts/validation/run_threshold_search.sh <path_to_checkpoint.pt>
#
# Example:
#   bash scripts/validation/run_threshold_search.sh finetune_logs/path_sam2_focal-2025-11-27_12-00-00/checkpoints/checkpoint_40.pt
#

set -e

CHECKPOINT=$1
OUTPUT_CONFIG="configs/optimal_thresholds.json"

if [ -z "$CHECKPOINT" ]; then
    echo "Usage: $0 <path_to_checkpoint.pt>"
    exit 1
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found at '$CHECKPOINT'"
    exit 1
fi

echo "=============================================="
echo "Optimal Threshold Search"
echo "=============================================="
echo "Checkpoint: $CHECKPOINT"
echo "Output:     $OUTPUT_CONFIG"
echo "Split:      val"
echo "=============================================="

python scripts/validation/find_optimal_thresholds.py \
    --checkpoint "$CHECKPOINT" \
    --model_cfg "sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml" \
    --split val \
    --output "$OUTPUT_CONFIG" \
    --thresholds 0.2 0.3 0.4 0.5 0.6 0.7 0.8

echo "✅ Optimal thresholds saved to $OUTPUT_CONFIG"
