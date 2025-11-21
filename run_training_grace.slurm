#!/bin/bash

#------------------------------------------------------------------
# Corrected Slurm job script for fine-tuning on TAMU HPRC Grace
#------------------------------------------------------------------

# --- SLURM Directives ---
#SBATCH --job-name=sam_finetune_best
#SBATCH --output=logs/sam_finetune_%j.out
#SBATCH --error=logs/sam_finetune_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8                # Adjusted from 32 for a more standard request
#SBATCH --mem=64G                        # Adjusted from 128G for a more standard request
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
#SBATCH --time=1-12:00:00

# --- Environment Setup ---
echo "Setting up the environment..."

# It's good practice to purge all modules to start with a clean environment
module purge

# Load the necessary module for deep learning
# This is the recommended way to load the DL stack on Grace
module load DL/2023.01

# Activate the conda environment
# Using an absolute path with $SCRATCH is a robust way to do this
source $SCRATCH/vfm_env/bin/activate

# Navigate to the project directory
cd $SCRATCH/vfm_project

# Ensure the log directory exists
mkdir -p logs

# Set the PYTHONPATH to include the project and sam2 directories
# This is a good practice to avoid import errors
export PYTHONPATH="$SCRATCH/vfm_project:$SCRATCH/vfm_project/sam2:$PYTHONPATH"
echo "PYTHONPATH set to: $PYTHONPATH"

# --- Job Execution ---
# The user-provided training script and arguments have been corrected to match the project structure
echo "Starting SAM2 finetuning with 'bcss_finetune_best' experiment..."
python src/run_finetuning.py experiment=bcss_finetune_best
echo "Finetuning completed."
