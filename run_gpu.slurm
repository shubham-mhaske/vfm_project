#!/bin/bash
#SBATCH --job-name=sam_finetune
#SBATCH --output=logs/sam_finetune_%j.out
#SBATCH --error=logs/sam_finetune_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G              # SAFE FOR YOUR TRAINING
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu         # Explicitly specify GPU partition
#SBATCH --time=1-00:00:00       # 1 day (adjust as needed)

module restore dl
source $SCRATCH/vfm_env/bin/activate
cd $SCRATCH/vfm_project
export PYTHONPATH="$SCRATCH/vfm_project/sam2:$SCRATCH/vfm_project:$PYTHONPATH"

echo "Starting SAM fine-tuning..."
python src/train_sam.py -c bcss_finetune
echo "Fine-tuning script finished."
