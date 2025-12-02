#!/bin/bash
#SBATCH --job-name=ensemble
#SBATCH --time=00:45:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --gres=gpu:a100:2
#SBATCH --partition=gpu
#SBATCH --output=slurm_logs/ensemble_%j.out
#SBATCH --error=slurm_logs/ensemble_%j.err

echo "============================================================"
echo "ENSEMBLE CLIP + PLIP"
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
export PLIP_MODEL_PATH="$SCRATCH/plip_model/plip_model"

python -u src/test_ensemble.py

echo "============================================================"
echo "Completed: $(date)"
echo "============================================================"
