#!/bin/bash
#SBATCH --job-name=eval_segmentation
#SBATCH --output=results_segmentation/eval_segmentation_%j.log
#SBATCH --error=results_segmentation/eval_segmentation_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=shubhammhaske@tamu.edu

# Activate your environment
source ~/.bashrc
conda activate vfm_project

# Move to project root
cd /scratch/user/shubhammhaske/vfm_project

# Run evaluation
python src/evaluate_segmentation.py \
  --checkpoint finetune_logs_improved/checkpoints/checkpoint_82.pt \
  --model_cfg sam2/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml \
  --image_dir data/bcss/images \
  --mask_dir data/bcss/masks \
  --split test \
  --prompt_type box \
  --use_neg_points \
  --output_dir results_segmentation
