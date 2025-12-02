#!/bin/bash
#SBATCH --job-name=multiscale
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --gres=gpu:a100:2
#SBATCH --partition=gpu
#SBATCH --output=slurm_logs/multiscale_%j.out
#SBATCH --error=slurm_logs/multiscale_%j.err

echo "============================================================"
echo "MULTI-SCALE CLIP CLASSIFICATION"
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

python -u src/test_multiscale.py

echo "============================================================"
echo "Completed: $(date)"
echo "============================================================"
