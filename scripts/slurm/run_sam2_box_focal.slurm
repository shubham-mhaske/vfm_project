#!/bin/bash

#------------------------------------------------------------------
# SLURM script for SAM2 Box+Focal experiment
# Goal: Test box prompting + focal loss for class imbalance
#------------------------------------------------------------------

# --- SLURM Directives ---
#SBATCH --job-name=sam2_box_focal
#SBATCH --output=slurm_logs/sam2_box_focal_%j.out
#SBATCH --error=slurm_logs/sam2_box_focal_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16               # More CPUs for data loading
#SBATCH --mem=128G                       # More memory for larger batch caching
#SBATCH --gres=gpu:a100:1                # 1 A100 is sufficient for this model
#SBATCH --partition=gpu
#SBATCH --time=6:00:00                   # 50 epochs

# --- Environment Setup ---
echo "=========================================="
echo "SAM2 Box+Focal Experiment"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

module purge
module restore dl

source $SCRATCH/vfm_env/bin/activate
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd $SCRATCH/vfm_project
mkdir -p slurm_logs

export PYTHONPATH="$SCRATCH/vfm_project:$SCRATCH/vfm_project/sam2:$PYTHONPATH"

# --- Job Execution ---
echo "Starting SAM2 Box+Focal finetuning..."
echo "Config: sam2_box_focal"
echo "Key changes:"
echo "  - Box prompts only (no points)"
echo "  - FocalDiceLoss for class imbalance"
echo "  - Class weights: blood_vessel 67.6x"
echo "  - LR: 3e-5"
echo ""

python src/run_finetuning.py experiment=sam2_box_focal

echo ""
echo "Training completed at $(date)"
echo "=========================================="
