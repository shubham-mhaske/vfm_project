#!/bin/bash

#------------------------------------------------------------------
# SLURM script for SAM2 LoRA-Light finetuning experiment
# Goal: Beat zeroshot (0.517) with minimal adaptation
#------------------------------------------------------------------

# --- SLURM Directives ---
#SBATCH --job-name=sam2_lora_light
#SBATCH --output=slurm_logs/sam2_lora_light_%j.out
#SBATCH --error=slurm_logs/sam2_lora_light_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16               # More CPUs for data loading
#SBATCH --mem=128G                       # More memory for larger batch caching
#SBATCH --gres=gpu:a100:1                # 1 A100 is sufficient for this model
#SBATCH --partition=gpu
#SBATCH --time=4:00:00                   # Short training (30 epochs)

# --- Environment Setup ---
echo "=========================================="
echo "SAM2 LoRA-Light Experiment"
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
echo "Starting SAM2 LoRA-Light finetuning..."
echo "Config: sam2_lora_light"
echo "Key changes vs v2:"
echo "  - LR: 5e-6 (12x smaller than v2)"
echo "  - Epochs: 30 (shorter)"
echo "  - Minimal augmentation"
echo "  - Dice-weighted loss"
echo ""

python src/run_finetuning.py experiment=sam2_lora_light

echo ""
echo "Training completed at $(date)"
echo "=========================================="
