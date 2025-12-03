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

# Verify environment
echo ""
echo "Environment Info:"
echo "  Python: $(which python)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'NOT FOUND')"
echo "  NumPy: $(python -c 'import numpy; print(numpy.__version__)' 2>/dev/null || echo 'NOT FOUND')"
echo "  CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'UNKNOWN')"
python -c 'import torch; print("  GPU:", torch.cuda.get_device_name(0))' 2>/dev/null || true
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
