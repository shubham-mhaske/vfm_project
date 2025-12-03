#!/bin/bash
#SBATCH --job-name=complete_metrics
#SBATCH --output=slurm_logs/complete_metrics-%j.out
#SBATCH --error=slurm_logs/complete_metrics-%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu

# ============================================================
# Comprehensive Metrics Collection Script
# ============================================================
# This script runs all experiments and collects complete metrics:
# - SAM2 Segmentation: Dice, IoU (6 prompt configurations)
# - CLIP Classification: Accuracy, Precision, Recall, F1 (8 prompts)
# - MedSAM Segmentation: Dice, IoU (with/without TTA)
#
# Expected runtime: ~4-6 hours
# Output: results/complete_metrics/
# ============================================================

echo "============================================================"
echo "JOB INFO"
echo "============================================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Job Name:      $SLURM_JOB_NAME"
echo "Node:          $SLURM_NODELIST"
echo "Start Time:    $(date)"
echo "============================================================"

# Navigate to project root
cd $SLURM_SUBMIT_DIR
cd ..  # Move from scripts/slurm to project root

# Create log directory
mkdir -p slurm_logs

# Load required modules
echo ""
echo "Loading modules..."
module purge
module load GCC/11.3.0
module load CUDA/11.7.0
module load Python/3.10.4

# Activate conda environment
echo "Activating conda environment..."
source ~/.bashrc
conda activate vfm

# Verify environment
echo ""
echo "Environment Info:"
echo "  Python: $(which python)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
if python -c 'import torch; print(torch.cuda.is_available())' | grep -q "True"; then
    echo "  GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
fi
echo ""

# Create output directory
mkdir -p results/complete_metrics

# Run the comprehensive metrics collection
echo "============================================================"
echo "STARTING COMPREHENSIVE METRICS COLLECTION"
echo "============================================================"
echo ""

python scripts/collect_all_metrics.py \
    --gpu 0

EXIT_CODE=$?

echo ""
echo "============================================================"
echo "JOB COMPLETE"
echo "============================================================"
echo "End Time:      $(date)"
echo "Exit Code:     $EXIT_CODE"
echo "Results Dir:   results/complete_metrics/"
echo "============================================================"

# List output files
if [ -d "results/complete_metrics" ]; then
    echo ""
    echo "Generated files:"
    ls -la results/complete_metrics/
fi

exit $EXIT_CODE
