#!/bin/bash
#SBATCH --job-name=full_pipeline
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
#SBATCH --output=slurm_logs/full_pipeline_%j.out
#SBATCH --error=slurm_logs/full_pipeline_%j.err

# ============================================================
# Full Pipeline Evaluation: SAM2 + CLIP
# ============================================================
# Evaluates all configurations with:
# - Segmentation: Dice and IoU
# - Classification: CLIP accuracy
# ============================================================

echo "============================================================"
echo "SLURM Job: Full Pipeline Evaluation (SAM2 + CLIP)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Started: $(date)"
echo "============================================================"

# Navigate to project root
PROJECT_ROOT="/scratch/user/$(whoami)/vfm_project"
cd "$PROJECT_ROOT" || { echo "ERROR: Project not found at $PROJECT_ROOT"; exit 1; }

# Create log directory
mkdir -p slurm_logs

# Load modules and activate environment
module purge
module restore dl

# Activate virtual environment
source $SCRATCH/vfm_env/bin/activate
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Set PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/sam2:$PYTHONPATH"

echo ""
echo "Running full pipeline evaluation..."
echo "Configs: baseline, refine2, refine3, TTA, TTA+refine"
echo "Metrics: Dice, IoU, CLIP Classification Accuracy"
echo ""

python src/evaluate_full_pipeline.py \
    --model_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
    --checkpoint sam2/checkpoints/sam2.1_hiera_large.pt \
    --clip_prompts configs/prompts/hard_coded_prompts_v2.json \
    --split test \
    --output_dir results/full_pipeline_eval

echo ""
echo "============================================================"
echo "Completed: $(date)"
echo "============================================================"

# Display results summary
if [ -f "results/full_pipeline_eval/summary.csv" ]; then
    echo ""
    echo "Results Summary:"
    cat results/full_pipeline_eval/summary.csv
fi
