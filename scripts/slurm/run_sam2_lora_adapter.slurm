#!/bin/bash

#------------------------------------------------------------------
# SLURM script for SAM2 LoRA Adapter training
# Uses new train_with_lora.py with proper LoRA implementation
# Goal: Add ~0.9% trainable params while keeping SAM2 frozen
#------------------------------------------------------------------

# --- SLURM Directives ---
#SBATCH --job-name=sam2_lora_adapter
#SBATCH --output=slurm_logs/sam2_lora_adapter_%j.out
#SBATCH --error=slurm_logs/sam2_lora_adapter_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
#SBATCH --time=6:00:00

# --- Environment Setup ---
echo "=========================================="
echo "SAM2 LoRA Adapter Training"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

module purge
module restore dl

source $SCRATCH/vfm_env/bin/activate
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd $SCRATCH/vfm_project
mkdir -p slurm_logs
mkdir -p finetune_logs/lora

export PYTHONPATH="$SCRATCH/vfm_project:$SCRATCH/vfm_project/sam2:$PYTHONPATH"

# --- Configuration ---
LORA_RANK=8
LR=1e-4
EPOCHS=20
BATCH_SIZE=4
WARMUP_EPOCHS=2

echo "Configuration:"
echo "  LoRA Rank: $LORA_RANK"
echo "  Learning Rate: $LR"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Warmup Epochs: $WARMUP_EPOCHS"
echo "  Target Modules: image_encoder"
echo ""
echo "Key features:"
echo "  - Freezes 224M original SAM2 params"
echo "  - Adds ~2M trainable LoRA params (0.9%)"
echo "  - Uses Dice + BCE loss"
echo "  - Cosine LR schedule with warmup"
echo ""

# --- Job Execution ---
echo "Starting LoRA adapter training..."

python src/train_with_lora.py \
    --sam_model_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
    --sam_checkpoint sam2/checkpoints/sam2.1_hiera_large.pt \
    --lora_rank $LORA_RANK \
    --lora_dropout 0.1 \
    --target_modules image_encoder \
    --lr $LR \
    --min_lr 1e-6 \
    --weight_decay 0.01 \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --warmup_epochs $WARMUP_EPOCHS \
    --data_dir data/bcss \
    --num_workers 8 \
    --output_dir finetune_logs/lora \
    --save_every 5 \
    --eval_every 2 \
    --device cuda \
    --amp

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Training completed at $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

# --- Post-training: Run evaluation if training succeeded ---
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Training successful! Outputs saved to finetune_logs/lora/"
    echo "To evaluate, run:"
    echo "  python src/evaluate_segmentation.py --checkpoint finetune_logs/lora/<run_dir>/best_lora_weights.pt"
fi

exit $EXIT_CODE
