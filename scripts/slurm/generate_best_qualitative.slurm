#!/bin/bash
#SBATCH --job-name=best_qual
#SBATCH --output=slurm_logs/best_qualitative-%j.out
#SBATCH --error=slurm_logs/best_qualitative-%j.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu

# ============================================================
# Generate Best-Performing Qualitative Results
# This script re-runs evaluation to get per-image metrics,
# then selects and visualizes the best-performing samples.
# ============================================================

echo "=============================================="
echo "Job: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started: $(date)"
echo "=============================================="

# Load modules using saved collection (faster)
module restore dl

# Activate virtual environment from SCRATCH
source $SCRATCH/vfm_env/bin/activate

# Change to project directory in SCRATCH
cd $SCRATCH/vfm_project

# Create output directories
mkdir -p results/best_qualitative
mkdir -p results/figures/presentation
mkdir -p slurm_logs

# Check GPU
echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

# Set optimal PyTorch settings for A100
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run the script with correct paths for HPRC
echo "Running best qualitative generation..."
python scripts/analysis/generate_best_qualitative.py \
    --data-dir $SCRATCH/vfm_project/data/bcss \
    --sam2-dir $SCRATCH/vfm_project/sam2

echo ""
echo "=============================================="
echo "Finished: $(date)"
echo "=============================================="
echo ""
echo "Output files:"
ls -la results/best_qualitative/
echo ""
echo "Generated figures:"
ls -la results/figures/presentation/
