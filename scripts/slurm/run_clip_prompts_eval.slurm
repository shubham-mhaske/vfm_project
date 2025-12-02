#!/bin/bash
#SBATCH --job-name=clip_prompts
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
#SBATCH --output=slurm_logs/clip_prompts_%j.out
#SBATCH --error=slurm_logs/clip_prompts_%j.err

# ============================================================
# CLIP Prompt Comparison Evaluation
# ============================================================
# Tests all CLIP prompt configurations (exp1, exp3, exp5) with
# the best SAM2 settings (Box + Neg + TTA + Refinement)
# ============================================================

echo "============================================================"
echo "SLURM Job: CLIP Prompt Comparison"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Started: $(date)"
echo "============================================================"

PROJECT_ROOT="/scratch/user/$(whoami)/vfm_project"
cd "$PROJECT_ROOT" || { echo "ERROR: Project not found"; exit 1; }

mkdir -p slurm_logs

# Load environment
echo "[$(date '+%H:%M:%S')] Loading modules..."
module purge
module restore dl

echo "[$(date '+%H:%M:%S')] Activating virtual environment..."
source $SCRATCH/vfm_env/bin/activate
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/sam2:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export CLIP_MODEL_PATH="$SCRATCH/clip_model"

echo "[$(date '+%H:%M:%S')] Environment ready"
echo "[$(date '+%H:%M:%S')] CLIP model: $CLIP_MODEL_PATH"
echo ""

# Run evaluation
python -u src/evaluate_clip_prompts.py \
    --model_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
    --checkpoint sam2/checkpoints/sam2.1_hiera_large.pt \
    --split test \
    --output_dir results/clip_prompts_comparison \
    --use_tta \
    --use_refinement

echo ""
echo "============================================================"
echo "Completed: $(date)"
echo "============================================================"

# Show results
if [ -f "results/clip_prompts_comparison/summary.csv" ]; then
    echo ""
    echo "Results Summary:"
    cat results/clip_prompts_comparison/summary.csv
fi
