#!/bin/bash
# Commands for base_finetune_v2_stable experiment

# ============================================================================
# TRAINING
# ============================================================================

# Test config resolves
python src/run_finetuning.py experiment=base_finetune_v2_stable --cfg job --resolve

# Submit to SLURM
sbatch run_training_grace.slurm base_finetune_v2_stable

# Monitor job
squeue -u $USER
tail -f finetune_logs/base_finetune_v2_stable-*/logs/train.log

# Check tensorboard
tensorboard --logdir finetune_logs/base_finetune_v2_stable-*/tensorboard

# ============================================================================
# EVALUATION
# ============================================================================

# Validate all checkpoints (automated)
bash experiments/base_finetune_v2_stable/validate_all_checkpoints.sh

# Quick validation check during training (e.g., epoch 25)
python src/evaluate_segmentation.py \
  --checkpoint finetune_logs/base_finetune_v2_stable-*/checkpoints/checkpoint_25.pt \
  --model_cfg sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml \
  --image_dir data/bcss/images \
  --mask_dir data/bcss/masks \
  --split val \
  --prompt_type box \
  --use_neg_points \
  --print_only \
  --tqdm

# Evaluate specific checkpoint (print-only)
CHECKPOINT_EPOCH=100  # Change this
python src/evaluate_segmentation.py \
  --checkpoint finetune_logs/base_finetune_v2_stable-*/checkpoints/checkpoint_${CHECKPOINT_EPOCH}.pt \
  --model_cfg sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml \
  --image_dir data/bcss/images \
  --mask_dir data/bcss/masks \
  --split val \
  --prompt_type box \
  --use_neg_points \
  --print_only \
  --tqdm

# Test set evaluation (replace with best checkpoint after validation)
python src/evaluate_segmentation.py \
  --checkpoint finetune_logs/base_finetune_v2_stable-*/checkpoints/checkpoint_<BEST>.pt \
  --model_cfg sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml \
  --image_dir data/bcss/images \
  --mask_dir data/bcss/masks \
  --split test \
  --prompt_type box \
  --use_neg_points \
  --output_dir results_segmentation/base_finetune_v2_stable_test \
  --tqdm

# ============================================================================
# COMPARISON WITH V1
# ============================================================================

# Quick comparison (best checkpoints)
echo "=== v1 Best (epoch 100) ==="
python src/evaluate_segmentation.py \
  --checkpoint finetune_logs/base_finetune-2025-11-21_20-45-29/checkpoints/checkpoint_100.pt \
  --model_cfg sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml \
  --image_dir data/bcss/images --mask_dir data/bcss/masks \
  --split val --prompt_type box --use_neg_points --print_only

echo "=== v2 Best (TBD) ==="
python src/evaluate_segmentation.py \
  --checkpoint finetune_logs/base_finetune_v2_stable-*/checkpoints/checkpoint_<BEST>.pt \
  --model_cfg sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml \
  --image_dir data/bcss/images --mask_dir data/bcss/masks \
  --split val --prompt_type box --use_neg_points --print_only
