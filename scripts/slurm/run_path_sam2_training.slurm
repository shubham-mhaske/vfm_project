#!/bin/bash
#SBATCH --job-name=path_sam2
#SBATCH --output=slurm_logs/path_sam2_%j.out
#SBATCH --error=slurm_logs/path_sam2_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00

###############################################################################
# PATH-SAM2 TRAINING SCRIPT
# SAM2 + UNI Encoder Fusion for Histopathology Segmentation
#
# This implements Path-SAM2 architecture:
# - SAM2 Hiera-L encoder (frozen) - general visual features
# - UNI ViT-L encoder (frozen) - histopathology-specific features  
# - Dimension alignment fusion (trainable) - combines both encoders
# - SAM2 decoder (trainable) - adapts to BCSS task
#
# Expected improvement: Dice 0.42 -> 0.65-0.80 on BCSS
###############################################################################

echo "=============================================="
echo "PATH-SAM2 Training Job"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start: $(date)"
echo "=============================================="

# --- Environment Setup ---
module purge
module load GCC/12.2.0
module load CUDA/12.1.1
module load Anaconda3/2024.02-1

# Activate your conda environment
source activate vfm_env  # or: conda activate vfm_env

# Navigate to project directory
cd /scratch/user/$USER/vfm_project  # Adjust path as needed

# Verify GPU access
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

# --- Verify Prerequisites ---
echo "Checking prerequisites..."

# Check SAM2 checkpoint
SAM2_CKPT="sam2/checkpoints/sam2.1_hiera_large.pt"
if [ ! -f "$SAM2_CKPT" ]; then
    echo "ERROR: SAM2 checkpoint not found at $SAM2_CKPT"
    echo "Run: bash sam2/checkpoints/download_ckpts.sh"
    exit 1
fi
echo "✓ SAM2 checkpoint: $SAM2_CKPT"

# Check UNI checkpoint
UNI_CKPT="models/uni/pytorch_model.bin"
if [ ! -f "$UNI_CKPT" ]; then
    echo "WARNING: UNI checkpoint not found at $UNI_CKPT"
    echo "Attempting to download UNI weights..."
    python src/download_uni_weights.py
    
    if [ ! -f "$UNI_CKPT" ]; then
        echo "ERROR: Failed to download UNI weights"
        echo "You need to:"
        echo "  1. Accept terms at https://huggingface.co/MahmoodLab/UNI"
        echo "  2. Run: huggingface-cli login"
        echo "  3. Run: python src/download_uni_weights.py"
        exit 1
    fi
fi
echo "✓ UNI checkpoint: $UNI_CKPT"

# Check dataset
DATA_DIR="data/bcss"
if [ ! -d "$DATA_DIR/images" ] || [ ! -d "$DATA_DIR/masks" ]; then
    echo "ERROR: BCSS dataset not found at $DATA_DIR"
    exit 1
fi
IMAGE_COUNT=$(ls -1 "$DATA_DIR/images"/*.png 2>/dev/null | wc -l)
echo "✓ BCSS dataset: $IMAGE_COUNT images"

# --- Environment Variables ---
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/sam2"
export CUDA_VISIBLE_DEVICES=0,1

# PyTorch distributed settings
export MASTER_ADDR=localhost
export MASTER_PORT=$((12355 + SLURM_JOB_ID % 1000))
export WORLD_SIZE=2
export OMP_NUM_THREADS=8

# Enable TF32 for A100 performance
export NVIDIA_TF32_OVERRIDE=1

# --- Training Configuration ---
EXPERIMENT="path_sam2_uni_fusion"
OUTPUT_DIR="finetune_logs/${EXPERIMENT}-${SLURM_JOB_ID}"
DATA_ROOT="data/bcss"

mkdir -p "$OUTPUT_DIR"
mkdir -p slurm_logs

echo ""
echo "Training Configuration:"
echo "  Experiment: $EXPERIMENT"
echo "  Output: $OUTPUT_DIR"
echo "  Data: $DATA_ROOT"
echo "  GPUs: 2x A100"
echo ""

# --- Run Path-SAM2 Training ---
echo "Starting Path-SAM2 training..."
echo ""

# Use torchrun for distributed training
torchrun \
    --nproc_per_node=2 \
    --master_port=$MASTER_PORT \
    src/run_path_sam2_training.py \
    experiment=$EXPERIMENT \
    data_root=$DATA_ROOT \
    hydra.run.dir=$OUTPUT_DIR \
    scratch.train_batch_size=3 \
    scratch.num_train_workers=8 \
    scratch.num_epochs=50

EXIT_CODE=$?

echo ""
echo "=============================================="
echo "Training Complete"
echo "Exit Code: $EXIT_CODE"
echo "End: $(date)"
echo "Output: $OUTPUT_DIR"
echo "=============================================="

# --- Post-training Evaluation (Optional) ---
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Running evaluation on best checkpoint..."
    
    # Find best checkpoint
    BEST_CKPT=$(ls -t "$OUTPUT_DIR/checkpoints"/*.pt 2>/dev/null | head -1)
    
    if [ -n "$BEST_CKPT" ]; then
        echo "Best checkpoint: $BEST_CKPT"
        
        # Run evaluation
        python src/evaluation.py \
            --sam_model_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
            --sam_checkpoint "$BEST_CKPT" \
            --clip_prompts configs/prompts/hard_coded_prompts_v2.json \
            --output_dir "results/path_sam2_eval_${SLURM_JOB_ID}"
    fi
fi

exit $EXIT_CODE
