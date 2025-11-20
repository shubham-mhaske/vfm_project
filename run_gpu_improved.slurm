#!/bin/bash
#SBATCH --job-name=sam_finetune_improved
#SBATCH --output=logs/sam_finetune_improved_%j.out
#SBATCH --error=logs/sam_finetune_improved_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
#SBATCH --time=1-12:00:00

module restore dl
source $SCRATCH/vfm_env/bin/activate
cd $SCRATCH/vfm_project
export PYTHONPATH="$SCRATCH/vfm_project:$SCRATCH/vfm_project/sam2:$PYTHONPATH"

echo "Starting SAM2 finetuning with IMPROVED config (hiera_l, mixed prompts, augmentations)..."
python src/train_sam.py -c bcss_finetune_improved
echo "Finetuning completed."
