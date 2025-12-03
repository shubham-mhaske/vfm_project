#!/bin/bash
# Run prompt engineering experiments on zeroshot SAM2
# Tests: TTA, different prompts, negative points

set -e

# Change to project root
cd /scratch/user/shubhammhaske/vfm_project

echo "=============================================="
echo "SAM2 Prompt Engineering Experiments"
echo "=============================================="

# Base settings
# NOTE: model_cfg should be relative to sam2 package (e.g., "configs/sam2.1/sam2.1_hiera_l.yaml")
# The evaluate_segmentation.py script handles the path correctly
CHECKPOINT="sam2/checkpoints/sam2.1_hiera_large.pt"
MODEL_CFG="configs/sam2.1/sam2.1_hiera_l.yaml"
COMMON_ARGS="--model_cfg $MODEL_CFG --checkpoint $CHECKPOINT --split test --tqdm"

# Create results directory
RESULTS_BASE="results/prompt_experiments_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_BASE"

echo ""
echo "[1/6] Zeroshot with BOX prompt (baseline)..."
python src/evaluate_segmentation.py $COMMON_ARGS \
    --prompt_type box \
    --output_dir "$RESULTS_BASE/box_baseline"

echo ""
echo "[2/6] Zeroshot with CENTROID (point) prompt..."
python src/evaluate_segmentation.py $COMMON_ARGS \
    --prompt_type centroid \
    --output_dir "$RESULTS_BASE/centroid"

echo ""
echo "[3/6] Zeroshot with MULTI_POINT prompt..."
python src/evaluate_segmentation.py $COMMON_ARGS \
    --prompt_type multi_point \
    --output_dir "$RESULTS_BASE/multi_point"

echo ""
echo "[4/6] Zeroshot BOX + NEGATIVE POINTS..."
python src/evaluate_segmentation.py $COMMON_ARGS \
    --prompt_type box \
    --use_neg_points \
    --output_dir "$RESULTS_BASE/box_neg_points"

echo ""
echo "[5/6] Zeroshot BOX + TEST-TIME AUGMENTATION (TTA)..."
python src/evaluate_segmentation.py $COMMON_ARGS \
    --prompt_type box \
    --use_tta \
    --output_dir "$RESULTS_BASE/box_tta"

echo ""
echo "[6/6] Zeroshot BOX + NEG POINTS + TTA (full)..."
python src/evaluate_segmentation.py $COMMON_ARGS \
    --prompt_type box \
    --use_neg_points \
    --use_tta \
    --output_dir "$RESULTS_BASE/box_neg_tta"

echo ""
echo "=============================================="
echo "All experiments complete!"
echo "Results saved to: $RESULTS_BASE"
echo "=============================================="

# Print summary
echo ""
echo "RESULTS SUMMARY:"
echo "----------------"
for dir in "$RESULTS_BASE"/*/; do
    name=$(basename "$dir")
    if [ -f "$dir/metrics.json" ]; then
        dice=$(python -c "import json; d=json.load(open('$dir/metrics.json')); print(f\"{d.get('overall', 'N/A'):.4f}\")" 2>/dev/null || echo "N/A")
        printf "%-20s : Dice = %s\n" "$name" "$dice"
    fi
done
