#!/bin/bash
#SBATCH --job-name=best_combo
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
#SBATCH --output=slurm_logs/best_combo_%j.out
#SBATCH --error=slurm_logs/best_combo_%j.err

# Quick experiment: Best prompts + blur overlay
echo "============================================================"
echo "BEST COMBO: Fewshot Prompts + Blur Overlay"
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "============================================================"

PROJECT_ROOT="/scratch/user/$(whoami)/vfm_project"
cd "$PROJECT_ROOT" || exit 1
mkdir -p slurm_logs

module purge
module restore dl
source $SCRATCH/vfm_env/bin/activate
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/sam2:$PYTHONPATH"
export CLIP_MODEL_PATH="$SCRATCH/clip_model"

python -u src/test_best_combo.py

echo "Completed: $(date)"
