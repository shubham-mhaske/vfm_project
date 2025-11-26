#!/bin/bash
#SBATCH --job-name=sam2_v3_perclass
#SBATCH --output=finetune_logs/v3_perclass-%j.out
#SBATCH --error=finetune_logs/v3_perclass-%j.err
#SBATCH --time=8:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G

# SAM2 v3 Training: Per-class validation with H&E augmentation
# Expected improvements: 0.42 -> 0.60-0.70 Dice

echo "============================================"
echo "SAM2 v3 Training: Per-Class Focused"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo ""

# Load modules
module purge
module load Anaconda3/2024.02-1
module load CUDA/12.1.1
module load cuDNN/8.9.7.29-CUDA-12.1.1

# Activate conda environment
source activate vfm_project

# Navigate to project directory
cd $SLURM_SUBMIT_DIR
echo "Working directory: $(pwd)"
echo ""

# Print environment info
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo ""

# Print config summary
echo "============================================"
echo "Training Configuration: v3_perclass"
echo "============================================"
echo "Epochs: 40 (reduced from 100, early stopping expected)"
echo "Batch size: 6"
echo "Learning rate: 8e-5 (increased for gradual unfreezing)"
echo "Warmup: 20% (longer for stability)"
echo "Loss weights: mask=5, dice=2, iou=1, class=1"
echo "Augmentation: H&E stain variation (ColorJitter 0.3)"
echo "Encoder: Frozen initially (gradual unfreezing strategy)"
echo "Expected: Dice 0.60-0.70 (vs v2: 0.42)"
echo ""

# Set data root
export DATA_ROOT="$SLURM_SUBMIT_DIR/data/bcss"

# Run training
echo "============================================"
echo "Starting Training..."
echo "============================================"
python src/run_finetuning.py \
    experiment=base_finetune_v3_perclass \
    data_root=$DATA_ROOT \
    hydra.run.dir=finetune_logs/base_finetune_v3_perclass-$(date +%Y-%m-%d_%H-%M-%S)

EXIT_CODE=$?
echo ""
echo "============================================"
echo "Training Complete!"
echo "============================================"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo ""

# If training successful, run per-class validation
if [ $EXIT_CODE -eq 0 ]; then
    echo "Running per-class validation on best checkpoint..."
    
    # Find best checkpoint (usually around epoch 15-25)
    CHECKPOINT_DIR=$(ls -dt finetune_logs/base_finetune_v3_perclass-* 2>/dev/null | head -1)
    
    if [ -d "$CHECKPOINT_DIR/checkpoints" ]; then
        # Find latest checkpoint as "best" (improve this with actual best selection)
        BEST_CHECKPOINT=$(ls -t $CHECKPOINT_DIR/checkpoints/checkpoint_*.pt 2>/dev/null | head -1)
        
        if [ -f "$BEST_CHECKPOINT" ]; then
            echo "Validating checkpoint: $BEST_CHECKPOINT"
            
            python scripts/validation/validate_perclass.py \
                --checkpoint $BEST_CHECKPOINT \
                --model_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
                --split val \
                --output $CHECKPOINT_DIR/validation_results.json
            
            echo "Validation results saved to: $CHECKPOINT_DIR/validation_results.json"
        else
            echo "Warning: No checkpoint found in $CHECKPOINT_DIR/checkpoints/"
        fi
    else
        echo "Warning: Checkpoint directory not found: $CHECKPOINT_DIR/checkpoints"
    fi
else
    echo "Training failed with exit code $EXIT_CODE"
    echo "Check logs for errors"
fi

echo ""
echo "============================================"
echo "Next Steps:"
echo "============================================"
echo "1. Check training logs: finetune_logs/v3_perclass-$SLURM_JOB_ID.out"
echo "2. View tensorboard: tensorboard --logdir=finetune_logs/base_finetune_v3_perclass-*/tensorboard"
echo "3. Run full evaluation:"
echo "   python src/evaluation.py \\"
echo "     --sam_checkpoint [BEST_CHECKPOINT] \\"
echo "     --sam_model_cfg configs/sam2.1/sam2.1_hiera_l.yaml \\"
echo "     --clip_prompts configs/prompts/hard_coded_prompts_v2.json \\"
echo "     --output_dir results/v3_evaluation"
echo "4. Compare with v2: Dice 0.42 -> target 0.60-0.70"
echo ""
