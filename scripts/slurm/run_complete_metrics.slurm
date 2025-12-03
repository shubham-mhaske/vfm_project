#!/bin/bash
#SBATCH --job-name=complete_metrics
#SBATCH --output=slurm_logs/complete_metrics-%j.out
#SBATCH --error=slurm_logs/complete_metrics-%j.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --gres=gpu:a100:2
#SBATCH --partition=gpu

# ============================================================
# Comprehensive Metrics Collection Script
# ============================================================
# This script runs all experiments and collects complete metrics:
# - SAM2 Segmentation: Dice, IoU (6 prompt configurations)
# - CLIP Classification: Accuracy, Precision, Recall, F1 (8 prompts)
# - MedSAM Segmentation: Dice, IoU (with/without TTA)
#
# Resources: 48 CPUs, 256GB RAM, 2x A100 GPUs
# Expected runtime: ~2-3 hours (optimized)
# Output: results/complete_metrics/
# ============================================================

echo "============================================================"
echo "JOB INFO"
echo "============================================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Job Name:      $SLURM_JOB_NAME"
echo "Node:          $SLURM_NODELIST"
echo "CPUs:          $SLURM_CPUS_PER_TASK"
echo "Memory:        256GB"
echo "GPUs:          2x A100"
echo "Start Time:    $(date)"
echo "============================================================"

# Navigate to project root (hardcoded path like other working scripts)
PROJECT_ROOT="/scratch/user/$(whoami)/vfm_project"
cd "$PROJECT_ROOT" || { echo "ERROR: Project not found at $PROJECT_ROOT"; exit 1; }

echo "Project Root:  $PROJECT_ROOT"
echo "Working Dir:   $(pwd)"

# Create directories
mkdir -p slurm_logs
mkdir -p results/complete_metrics

# Load modules and activate environment (use saved module collection like other scripts)
echo ""
echo "Loading modules..."
module purge
module restore dl

# Activate virtual environment (use $SCRATCH path like other working scripts)
echo "Activating virtual environment..."
source $SCRATCH/vfm_env/bin/activate
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Set PYTHONPATH for imports
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src:$PROJECT_ROOT/sam2:$PROJECT_ROOT/MedSAM:$PYTHONPATH"

# Optimize for multi-threading
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# PyTorch optimizations
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Verify environment
echo ""
echo "Environment Info:"
echo "  Python: $(which python)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'NOT FOUND')"
echo "  NumPy: $(python -c 'import numpy; print(numpy.__version__)' 2>/dev/null || echo 'NOT FOUND')"
echo "  CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'UNKNOWN')"
echo "  GPU count: $(python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 'UNKNOWN')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo "  Threads: $OMP_NUM_THREADS"
echo ""

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
