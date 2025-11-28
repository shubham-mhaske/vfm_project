#!/bin/bash
#SBATCH --job-name=eval_lora
#SBATCH --output=slurm_logs/eval_lora_%j.out
#SBATCH --error=slurm_logs/eval_lora_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=1:00:00

# ============================================================================
# Evaluate SAM2 LoRA Adapter on BCSS Test Set
# 
# This script evaluates the trained LoRA weights using the FULL SAM2 predictor
# with proper prompts (not the simplified backbone-only forward pass from training).
# This ensures fair comparison with zeroshot baseline (0.517 Dice).
#
# Usage:
#   sbatch scripts/slurm/eval_lora_adapter.slurm <lora_weights_path> [output_dir]
#
# Example:
#   sbatch scripts/slurm/eval_lora_adapter.slurm \
#       finetune_logs/lora/lora_r8_20251127_204826/best_lora_weights.pt \
#       results/lora_r8_eval
# ============================================================================

echo "========================================"
echo "SAM2 LoRA Evaluation Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started: $(date)"
echo "========================================"

# Get arguments
LORA_WEIGHTS=${1:-"finetune_logs/lora/lora_r8_20251127_204826/best_lora_weights.pt"}
OUTPUT_DIR=${2:-"results/lora_r8_eval_$(date +%Y%m%d_%H%M%S)"}

echo "LoRA weights: $LORA_WEIGHTS"
echo "Output dir: $OUTPUT_DIR"

# Navigate to project root
cd /scratch/user/shubhammhaske/vfm_project
echo "Working directory: $(pwd)"

# Activate conda environment
source ~/.bashrc
conda activate vfm

# Verify environment
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
if python -c 'import torch; print(torch.cuda.is_available())' | grep -q "True"; then
    echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p slurm_logs

echo ""
echo "========================================"
echo "Starting LoRA Evaluation"
echo "========================================"

# Run evaluation with full SAM2 predictor
python src/evaluate_lora.py \
    --lora_weights "$LORA_WEIGHTS" \
    --lora_rank 8 \
    --sam_model_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
    --sam_checkpoint sam2/checkpoints/sam2.1_hiera_large.pt \
    --split test \
    --prompt_type box \
    --output_dir "$OUTPUT_DIR" \
    --tqdm

EVAL_EXIT_CODE=$?

echo ""
echo "========================================"
echo "Evaluation Complete"
echo "Exit code: $EVAL_EXIT_CODE"
echo "Finished: $(date)"
echo "========================================"

# Print results summary if available
if [ -f "$OUTPUT_DIR/metrics.json" ]; then
    echo ""
    echo "Results Summary:"
    cat "$OUTPUT_DIR/metrics.json"
fi

exit $EVAL_EXIT_CODE
