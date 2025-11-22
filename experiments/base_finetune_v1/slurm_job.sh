#!/bin/bash
#SBATCH --job-name=base_finetune_v1
#SBATCH --output=slurm_logs/base_finetune_v1_%j.out
#SBATCH --error=slurm_logs/base_finetune_v1_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB

# Original job info
# Job ID: (not recorded)
# Submitted: 2025-11-21
# Runtime directory: finetune_logs/base_finetune-2025-11-21_20-45-29/

module restore dl 

source $SCRATCH/vfm_env/bin/activate

cd $SCRATCH/vfm_project

# Set data root
export data_root=/scratch/user/shubhammhaske/vfm_project/data/bcss

# Run training
python src/run_finetuning.py experiment=base_finetune

echo "Training completed at $(date)"
