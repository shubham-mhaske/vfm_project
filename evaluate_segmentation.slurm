#!/bin/bash
#SBATCH --job-name=eval_segmentation
#SBATCH --output=results_segmentation/eval_segmentation_%j.log
#SBATCH --error=results_segmentation/eval_segmentation_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
#SBATCH --time=1-12:00:00

# Activate your environment

module restore dl
source $SCRATCH/vfm_env/bin/activate
cd $SCRATCH/vfm_project
export PYTHONPATH="$SCRATCH/vfm_project:$SCRATCH/vfm_project/sam2:$PYTHONPATH"

# Run evaluation
echo "Starting segmentation evaluation..."
python -u src/evaluate_segmentation.py \
  --checkpoint finetune_logs_improved/checkpoints/checkpoint_82.pt \
  --model_cfg sam2/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml \
  --image_dir data/bcss/images \
  --mask_dir data/bcss/masks \
  --split test \
  --prompt_type box \
  --use_neg_points \
  --output_dir results_segmentation \
  --verbose \
  --max_samples 25
echo "Segmentation evaluation completed."
