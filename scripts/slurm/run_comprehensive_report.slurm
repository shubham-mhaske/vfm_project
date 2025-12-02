#!/bin/bash
#SBATCH --job-name=comp_report
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
#SBATCH --output=slurm_logs/comp_report_%j.out
#SBATCH --error=slurm_logs/comp_report_%j.err

# ============================================================
# Comprehensive Metrics Report Generator
# ============================================================
# Generates all metrics for paper/presentation:
# - Segmentation: Dice, IoU, per-class, confidence intervals
# - Classification: Accuracy, Precision, Recall, F1, Confusion Matrix
# - LaTeX tables and figures
# ============================================================

echo "============================================================"
echo "SLURM Job: Comprehensive Report Generator"
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
echo ""

# Run comprehensive report generator
python -u src/generate_comprehensive_report.py \
    --model_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
    --checkpoint sam2/checkpoints/sam2.1_hiera_large.pt \
    --clip_prompts configs/prompts/hard_coded_prompts_v2.json \
    --split test \
    --output_dir results/comprehensive_report

echo ""
echo "============================================================"
echo "Completed: $(date)"
echo "============================================================"

# List generated files
echo ""
echo "Generated files:"
ls -la results/comprehensive_report/
