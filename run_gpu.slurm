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

# Load required modules
module restore dl

# Activate Python virtual environment
# IMPORTANT: Replace with the actual path to your virtual environment
source $SCRATCH/venvs/vfm_env/bin/activate

# Change to the project directory on scratch
cd $SCRATCH/vfm_project

# Run the training script
# The train.py script uses Hydra and finds the config by name.
# The output directory will be created automatically inside the project directory.
python src/train_sam.py -c bcss_finetune
