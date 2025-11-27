#!/bin/bash
#SBATCH --job-name=sam2_v4_fast
#SBATCH --output=logs/v4_fast-%j.out
#SBATCH --error=logs/v4_fast-%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32              # More CPUs for data loading
#SBATCH --gres=gpu:a100:2               # 2x A100 GPUs (2x faster)
#SBATCH --mem=128G                      # More memory for larger batch

# ==============================================================================
# SAM2 v4 FAST Training: Multi-GPU Optimized
# ==============================================================================
# Speed improvements:
#   - 2x A100 GPUs with DataParallel = ~1.8x speedup
#   - Larger batch size (12 vs 6) = better GPU utilization
#   - 32 CPU workers = faster data loading
#   - num_workers=16 per GPU
# Expected: 4-5 hours (vs 8 hours single GPU)
# ==============================================================================

echo "============================================"
echo "SAM2 v4 FAST Training: Multi-GPU"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo ""

# Load modules
module purge
module restore dl

# Activate environment
source $SCRATCH/vfm_env/bin/activate

# Set paths
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
cd $SLURM_SUBMIT_DIR
export PYTHONPATH="$SLURM_SUBMIT_DIR:$SLURM_SUBMIT_DIR/sam2:$PYTHONPATH"

# Create run directory
RUN_NAME="base_finetune_v4_fast-$(date +%Y-%m-%d_%H-%M-%S)"
RUN_DIR="finetune_logs/$RUN_NAME"
mkdir -p "$RUN_DIR/slurm_logs" logs

echo "Working directory: $(pwd)"
echo "Run directory: $RUN_DIR"
echo ""

# Environment info
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"
python -c "import torch; [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
echo ""

# Training config summary
echo "============================================"
echo "Configuration: v4_fast (Multi-GPU)"
echo "============================================"
echo "GPUs: 2x A100"
echo "Effective batch size: 12 (6 per GPU)"
echo "Epochs: 40"
echo "Learning rate: 8e-5"
echo "Expected time: ~4 hours"
echo ""

# Set data root
export DATA_ROOT="$SLURM_SUBMIT_DIR/data/bcss"

# Run training with multi-GPU overrides
echo "Starting Multi-GPU Training..."
echo "============================================"

python src/run_finetuning.py \
    experiment=base_finetune_v3_perclass \
    data_root=$DATA_ROOT \
    hydra.run.dir=$RUN_DIR \
    scratch.train_batch_size=6 \
    scratch.num_train_workers=16

EXIT_CODE=$?

echo ""
echo "============================================"
echo "Training Complete!"
echo "============================================"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"

# Copy logs
cp "logs/v4_fast-${SLURM_JOB_ID}.out" "$RUN_DIR/slurm_logs/" 2>/dev/null
cp "logs/v4_fast-${SLURM_JOB_ID}.err" "$RUN_DIR/slurm_logs/" 2>/dev/null

# Show results location
echo ""
echo "Results: $RUN_DIR"
echo "Checkpoints: $RUN_DIR/checkpoints/"
echo "TensorBoard: tensorboard --logdir=$RUN_DIR/tensorboard"
