#!/bin/bash
#SBATCH --job-name=base_finetune_v2_stable
#SBATCH --output=slurm_logs/base_finetune_v2_stable_%j.out
#SBATCH --error=slurm_logs/base_finetune_v2_stable_%j.err
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB

# Experiment: base_finetune_v2_stable
# Changes: LR 6e-5, warmup 175 steps, batch size 6 (fits A100 40GB),
#          balanced loss (5:2:1:1), lower weight decay (0.01), 100 epochs,
#          FROZEN image encoder (trains only prompt/mask/memory, ~20-30M params)
# Expected time: ~8-10 hours (faster due to frozen encoder)


module restore dl 

source $SCRATCH/vfm_env/bin/activate

cd $SCRATCH/vfm_project

# Set data root
export data_root=/scratch/user/shubhammhaske/vfm_project/data/bcss

# Run training
python src/run_finetuning.py experiment=base_finetune_v2_stable

echo "Training completed at $(date)"
