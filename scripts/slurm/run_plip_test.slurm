#!/bin/bash
#SBATCH --job-name=plip_test
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
#SBATCH --output=slurm_logs/plip_test_%j.out
#SBATCH --error=slurm_logs/plip_test_%j.err

# Test PLIP (Pathology Language-Image Pretraining) for classification
echo "============================================================"
echo "PLIP TEST: Pathology-Specific CLIP"
echo "Expected: 38.9% (CLIP) → 55-65% (PLIP)"
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "============================================================"

PROJECT_ROOT="/scratch/user/$(whoami)/vfm_project"
cd "$PROJECT_ROOT" || exit 1
mkdir -p slurm_logs

module purge
module restore dl
source $SCRATCH/vfm_env/bin/activate

# Install transformers if not present (for PLIP)
pip install transformers --quiet 2>/dev/null

export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/sam2:$PYTHONPATH"
export HF_HOME="$SCRATCH/huggingface_cache"
mkdir -p "$HF_HOME"

python -u src/test_plip.py

echo "============================================================"
echo "Completed: $(date)"
echo "============================================================"
