#!/bin/bash
#SBATCH --job-name=base_finetune_v2_stable
#SBATCH --output=slurm_logs/base_finetune_v2_stable_%j.out
#SBATCH --error=slurm_logs/base_finetune_v2_stable_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB

# Experiment: base_finetune_v2_stable
# Changes: Higher LR (7e-5), warmup (125 steps), batch size 8 (2x faster),
#          balanced loss (5:2:1:1), lower weight decay (0.01), 100 epochs
# Expected time: ~10 hours (vs 16h for batch=4)

module purge
module load GCCcore/11.3.0 Python/3.10.4

source ~/.bashrc
conda activate vfm_env

cd /scratch/user/shubhammhaske/vfm_project

# Set data root
export data_root=/scratch/user/shubhammhaske/vfm_project/data/bcss

# Run training
python src/run_finetuning.py experiment=base_finetune_v2_stable

echo "Training completed at $(date)"
