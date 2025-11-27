#!/bin/bash
#SBATCH --job-name=path_sam2_ctp_optimized
#SBATCH --output=slurm_logs/path_sam2_ctp_optimized_%j.out
#SBATCH --error=slurm_logs/path_sam2_ctp_optimized_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00

###############################################################################
# PATH-SAM2 TRAINING WITH CTRANSPATH (OPTIMIZED)
# 
# This is an optimized version for faster training.
#
# Key Changes:
#   - AMP: bfloat16
#   - Fusion: 'concat' (simpler, faster)
#   - CTransPath: Frozen
#   - Batch Size: 6
#   - Workers: 12
#
# Expected improvement: 5-8x faster training, slight Dice gain
###############################################################################

echo "=============================================="
echo "Path-SAM2 + CTransPath Training (OPTIMIZED)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start: $(date)"
echo "=============================================="

# --- Environment Setup ---
module restore dl
source $SCRATCH/vfm_env/bin/activate

cd $SCRATCH/vfm_project

export PYTHONPATH="$SCRATCH/vfm_project:$SCRATCH/vfm_project/sam2:$PYTHONPATH"

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
    pip install -q gdown
    mkdir -p models/ctranspath
    gdown 1DoDx_70_TLj98gTf6YTXnu4tFhsFocDX -O "$CTP_CKPT"
    
    if [ ! -f "$CTP_CKPT" ]; then
        echo "ERROR: Failed to download CTransPath weights"
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
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=12
export NVIDIA_TF32_OVERRIDE=1

export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=12356 # Different port just in case

# --- Training Config ---
EXPERIMENT="path_sam2_ctranspath_optimized"
OUTPUT_DIR="finetune_logs/${EXPERIMENT}-${SLURM_JOB_ID}"
DATA_ROOT="data/bcss"

mkdir -p "$OUTPUT_DIR"
mkdir -p slurm_logs

echo ""
echo "Configuration:"
echo "  Experiment: $EXPERIMENT"
echo "  Output: $OUTPUT_DIR"
echo "  Fusion: concat (SAM2 256-dim + CTransPath 768-dim → 256-dim)"
echo "  GPU: Single A100 (WORLD_SIZE=1)"
echo ""

# --- Run Training ---
echo "Starting training..."

python src/run_path_sam2_ctranspath.py \
    experiment=$EXPERIMENT \
    data_root=$DATA_ROOT \
    hydra.run.dir=$OUTPUT_DIR

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
            --output_dir "results/path_sam2_ctp_optimized_eval_${SLURM_JOB_ID}"
    fi
fi

exit $EXIT_CODE
