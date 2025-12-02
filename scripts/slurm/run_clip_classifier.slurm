#!/bin/bash
#SBATCH --job-name=clip_classifier
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
#SBATCH --output=slurm_logs/clip_classifier_%j.out
#SBATCH --error=slurm_logs/clip_classifier_%j.err

# Train classifier on CLIP features
# Expected: 50-65% (vs 38.9% zero-shot)

echo "============================================================"
echo "CLIP FEATURE CLASSIFIER"
echo "Extract CLIP embeddings → Train LogisticRegression/MLP"
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

echo "CLIP model path: $CLIP_MODEL_PATH"

python -u src/train_clip_classifier.py

echo "============================================================"
echo "Completed: $(date)"
echo "============================================================"
