#!/bin/bash
#SBATCH --job-name=sam_finetune_best
#SBATCH --output=logs/sam_finetune_%j.out
#SBATCH --error=logs/sam_finetune_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
#SBATCH --time=2-00:00:00

module restore dl
source $SCRATCH/vfm_env/bin/activate
cd $SCRATCH/vfm_project
export PYTHONPATH="$SCRATCH/vfm_project:$SCRATCH/vfm_project/sam2:$PYTHONPATH"

echo "Starting SAM2 finetuning with best config (box+negatives prompts)..."
python src/train_sam.py -c bcss_finetune_best
echo "Finetuning completed."
