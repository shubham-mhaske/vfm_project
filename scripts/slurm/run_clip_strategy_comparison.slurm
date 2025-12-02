#!/bin/bash
#SBATCH --job-name=clip_strategy
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
#SBATCH --output=slurm_logs/clip_strategy_%j.out
#SBATCH --error=slurm_logs/clip_strategy_%j.err

# ============================================================
# CLIP Strategy Comparison
# ============================================================
# Compares different crop/overlay strategies for optimal
# SAM + CLIP integration
# ============================================================

echo "============================================================"
echo "CLIP Strategy Comparison"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "============================================================"

PROJECT_ROOT="/scratch/user/$(whoami)/vfm_project"
cd "$PROJECT_ROOT" || { echo "ERROR: Project not found"; exit 1; }

mkdir -p slurm_logs

# Load environment
echo "[$(date '+%H:%M:%S')] Loading modules..."
module purge
module restore dl

echo "[$(date '+%H:%M:%S')] Activating virtual environment..."
source $SCRATCH/vfm_env/bin/activate
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/sam2:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export CLIP_MODEL_PATH="$SCRATCH/clip_model"

echo "[$(date '+%H:%M:%S')] Environment ready"
echo ""

# Run strategy comparison
python -u src/compare_clip_strategies.py \
    --clip_prompts configs/prompts/hard_coded_prompts_v2.json \
    --split test \
    --output_dir results/clip_strategy_comparison

echo ""
echo "============================================================"
echo "Completed: $(date)"
echo "============================================================"
