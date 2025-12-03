#!/bin/bash
#SBATCH --job-name=best_qual
#SBATCH --output=slurm_logs/best_qualitative-%j.out
#SBATCH --error=slurm_logs/best_qualitative-%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
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

# Load modules
module purge
module load GCC/12.2.0
module load CUDA/12.0.0
module load Python/3.10.8

# Activate virtual environment
source ~/vfm_env/bin/activate

# Change to project directory
cd $SLURM_SUBMIT_DIR

# Create output directories
mkdir -p results/best_qualitative
mkdir -p results/figures/presentation
mkdir -p slurm_logs

# Check GPU
echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

# Run the script
echo "Running best qualitative generation..."
python scripts/analysis/generate_best_qualitative.py

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
