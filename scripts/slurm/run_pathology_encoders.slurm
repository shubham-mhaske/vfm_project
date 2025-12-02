#!/bin/bash
#SBATCH --job-name=path_encoders
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --gres=gpu:a100:2
#SBATCH --partition=gpu
#SBATCH --output=slurm_logs/path_encoders_%j.out
#SBATCH --error=slurm_logs/path_encoders_%j.err

echo "============================================================"
echo "PATHOLOGY ENCODERS: UNI vs CTransPath"
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "============================================================"

PROJECT_ROOT="/scratch/user/$(whoami)/vfm_project"
cd "$PROJECT_ROOT" || exit 1
mkdir -p slurm_logs

module purge
module restore dl
source $SCRATCH/vfm_env/bin/activate

# Install timm if needed
pip install timm --quiet 2>/dev/null

export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/sam2:$PYTHONPATH"
export CTRANSPATH_MODEL_PATH="$PROJECT_ROOT/models/ctranspath"
export UNI_MODEL_PATH="$PROJECT_ROOT/models/uni_model"

echo "CTransPath path: $CTRANSPATH_MODEL_PATH"
echo "UNI path: $UNI_MODEL_PATH"
ls -la "$CTRANSPATH_MODEL_PATH" 2>/dev/null || echo "CTransPath not found"
ls -la "$UNI_MODEL_PATH" 2>/dev/null || echo "UNI not found"

python -u src/test_pathology_encoders.py

echo "============================================================"
echo "Completed: $(date)"
echo "============================================================"
