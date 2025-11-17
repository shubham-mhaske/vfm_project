#!/bin/bash
#SBATCH --job-name=sam_training
#SBATCH --output=logs/sam_training_%j.out
#SBATCH --error=logs/sam_training_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

# Load required modules/environment
module restore dl

# Activate Python virtual environment (update path if needed)
source $SCRATCH/venvs/vfm_env/bin/activate

# Change to project directory
cd $SCRATCH/vfm_project

# Ensure log directory exists
mkdir -p logs

# Run training (use explicit config/output if possible)
python src/train_sam.py --config sam2/training/configs/bcss_finetune.yaml --output_dir $SCRATCH/vfm_project/results/exp_finetune_sam
