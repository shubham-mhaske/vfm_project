#!/bin/bash
# Commands for base_finetune_v1 experiment

# ============================================================================
# TRAINING
# ============================================================================

# Local test (before HPC submission)
python src/run_finetuning.py experiment=base_finetune --cfg job --resolve

# Submit to SLURM
sbatch run_training_grace.slurm base_finetune

# Monitor job
squeue -u $USER
tail -f finetune_logs/base_finetune-*/logs/train.log

# ============================================================================
# EVALUATION
# ============================================================================

# Quick validation (print-only, no saves)
python src/evaluate_segmentation.py \
  --checkpoint finetune_logs/base_finetune-2025-11-21_20-45-29/checkpoints/checkpoint_100.pt \
  --model_cfg sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml \
  --image_dir data/bcss/images \
  --mask_dir data/bcss/masks \
  --split val \
  --prompt_type box \
  --use_neg_points \
  --print_only \
  --tqdm

# Full validation with results saved
python src/evaluate_segmentation.py \
  --checkpoint finetune_logs/base_finetune-2025-11-21_20-45-29/checkpoints/checkpoint_100.pt \
  --model_cfg sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml \
  --image_dir data/bcss/images \
  --mask_dir data/bcss/masks \
  --split val \
  --prompt_type box \
  --use_neg_points \
  --output_dir results_segmentation/base_finetune_v1_val \
  --tqdm

# Test set evaluation (best checkpoint)
python src/evaluate_segmentation.py \
  --checkpoint finetune_logs/base_finetune-2025-11-21_20-45-29/checkpoints/checkpoint_100.pt \
  --model_cfg sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml \
  --image_dir data/bcss/images \
  --mask_dir data/bcss/masks \
  --split test \
  --prompt_type box \
  --use_neg_points \
  --output_dir results_segmentation/base_finetune_v1_test \
  --tqdm

# Validate all checkpoints
for epoch in 10 20 30 40 50 60 70 80 90 100 110; do
  echo "=== Epoch $epoch ==="
  python src/evaluate_segmentation.py \
    --checkpoint finetune_logs/base_finetune-2025-11-21_20-45-29/checkpoints/checkpoint_${epoch}.pt \
    --model_cfg sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml \
    --image_dir data/bcss/images --mask_dir data/bcss/masks \
    --split val --prompt_type box --use_neg_points --print_only
done
