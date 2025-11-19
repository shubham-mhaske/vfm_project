#!/bin/bash
#SBATCH --job-name=sam_finetune
#SBATCH --output=logs/sam_finetune_%j.out
#SBATCH --error=logs/sam_finetune_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

# Load required modules/environment
module restore dl

# Activate Python virtual environment
source $SCRATCH/vfm_env/bin/activate

# Change to the project directory on scratch
cd $SCRATCH/vfm_project

# Ensure both the SAM2 repo and project root are importable
export PYTHONPATH="$SCRATCH/vfm_project/sam2:$SCRATCH/vfm_project:$PYTHONPATH"

# Run the fine-tuning script
echo "Starting SAM fine-tuning..."
python src/train_sam.py -c bcss_finetune
echo "Fine-tuning script finished."
