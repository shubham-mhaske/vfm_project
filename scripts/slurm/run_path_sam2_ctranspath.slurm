#!/bin/bash
#SBATCH --job-name=path_sam2_ctp
#SBATCH --output=slurm_logs/path_sam2_ctp_%j.out
#SBATCH --error=slurm_logs/path_sam2_ctp_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00

###############################################################################
# PATH-SAM2 TRAINING WITH CTRANSPATH
# 
# Architecture:
#   SAM2 Hiera-L (256-dim) + CTransPath Swin (768-dim) → Fusion → Decoder
#
# CTransPath: Swin Transformer pretrained on 15M TCGA histopathology patches
# No access restrictions - fully open source!
#
# Expected improvement: Dice 0.42 → 0.60-0.75
###############################################################################

echo "=============================================="
echo "Path-SAM2 + CTransPath Training"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start: $(date)"
echo "=============================================="

# --- Environment Setup ---
module purge
module load GCC/12.2.0
module load CUDA/12.1.1
module load Anaconda3/2024.02-1

source activate vfm_env

cd /scratch/user/$USER/vfm_project

# GPU info
echo ""
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

# --- Check Prerequisites ---
echo "Checking prerequisites..."

# SAM2 checkpoint
SAM2_CKPT="sam2/checkpoints/sam2.1_hiera_large.pt"
if [ ! -f "$SAM2_CKPT" ]; then
    echo "ERROR: SAM2 checkpoint not found"
    echo "Run: bash sam2/checkpoints/download_ckpts.sh"
    exit 1
fi
echo "✓ SAM2: $SAM2_CKPT"

# CTransPath checkpoint
CTP_CKPT="models/ctranspath/ctranspath.pth"
if [ ! -f "$CTP_CKPT" ]; then
    echo "CTransPath weights not found. Downloading..."
    
    # Try gdown
    pip install -q gdown
    mkdir -p models/ctranspath
    gdown 1DoDx_70_TLj98gTf6YTXnu4tFhsFocDX -O "$CTP_CKPT"
    
    if [ ! -f "$CTP_CKPT" ]; then
        echo "ERROR: Failed to download CTransPath weights"
        echo "Manual download:"
        echo "  gdown 1DoDx_70_TLj98gTf6YTXnu4tFhsFocDX -O models/ctranspath/ctranspath.pth"
        exit 1
    fi
fi
echo "✓ CTransPath: $CTP_CKPT"

# Dataset
DATA_DIR="data/bcss"
IMAGE_COUNT=$(ls -1 "$DATA_DIR/images"/*.png 2>/dev/null | wc -l)
echo "✓ BCSS dataset: $IMAGE_COUNT images"

# --- Environment Variables ---
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/sam2"
export CUDA_VISIBLE_DEVICES=0,1
export MASTER_ADDR=localhost
export MASTER_PORT=$((12355 + SLURM_JOB_ID % 1000))
export WORLD_SIZE=2
export OMP_NUM_THREADS=8
export NVIDIA_TF32_OVERRIDE=1

# --- Training Config ---
EXPERIMENT="path_sam2_ctranspath"
OUTPUT_DIR="finetune_logs/${EXPERIMENT}-${SLURM_JOB_ID}"
DATA_ROOT="data/bcss"

mkdir -p "$OUTPUT_DIR"
mkdir -p slurm_logs

echo ""
echo "Configuration:"
echo "  Experiment: $EXPERIMENT"
echo "  Output: $OUTPUT_DIR"
echo "  Fusion: concat (SAM2 256-dim + CTransPath 768-dim → 256-dim)"
echo "  GPUs: 2x A100"
echo ""

# --- Run Training ---
echo "Starting training..."

torchrun \
    --nproc_per_node=2 \
    --master_port=$MASTER_PORT \
    src/run_path_sam2_ctranspath.py \
    experiment=$EXPERIMENT \
    data_root=$DATA_ROOT \
    hydra.run.dir=$OUTPUT_DIR \
    scratch.train_batch_size=3 \
    scratch.num_train_workers=8 \
    scratch.num_epochs=50

EXIT_CODE=$?

echo ""
echo "=============================================="
echo "Complete!"
echo "Exit: $EXIT_CODE"
echo "End: $(date)"
echo "Output: $OUTPUT_DIR"
echo "=============================================="

# --- Evaluation ---
if [ $EXIT_CODE -eq 0 ]; then
    BEST_CKPT=$(ls -t "$OUTPUT_DIR/checkpoints"/*.pt 2>/dev/null | head -1)
    
    if [ -n "$BEST_CKPT" ]; then
        echo ""
        echo "Running evaluation on: $BEST_CKPT"
        
        python src/evaluation.py \
            --sam_model_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
            --sam_checkpoint "$BEST_CKPT" \
            --clip_prompts configs/prompts/hard_coded_prompts_v2.json \
            --output_dir "results/path_sam2_ctp_eval_${SLURM_JOB_ID}"
    fi
fi

exit $EXIT_CODE
