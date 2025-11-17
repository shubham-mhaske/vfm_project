#!/bin/bash
#SBATCH --job-name=sam_finetune
#SBATCH --output=logs/logs/sam_finetune_%j.out
#SBATCH --error=logs/logs/sam_finetune_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

# Load required modules/environment
# This command restores the module environment saved with 'module save dl'.
# Ensure you have a saved 'dl' environment or replace with 'module load <modules>'.
module restore dl

# Activate Python virtual environment
# IMPORTANT: Replace with the correct path to your virtual environment.
source $SCRATCH/vfm_env/bin/activate

# Change to the project directory on scratch
cd $SCRATCH/vfm_project

# Ensure both the SAM2 repo and project root are importable
# - 'sam2' provides the installed package (configs, modeling)
# - 'training' is a top-level package inside the SAM2 repo
# - 'src' provides BCSSRawDataset for finetuning
export PYTHONPATH="$SCRATCH/vfm_project/sam2:$SCRATCH/vfm_project:$PYTHONPATH"

# Run the fine-tuning script
# The '-c' argument specifies the configuration name from sam2/training/configs
echo "Starting SAM fine-tuning..."
python src/train_sam.py -c bcss_finetune
echo "Fine-tuning script finished."
