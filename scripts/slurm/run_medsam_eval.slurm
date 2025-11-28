#!/bin/bash
#SBATCH --job-name=medsam_eval
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
#SBATCH --output=slurm_logs/medsam_eval_%j.out
#SBATCH --error=slurm_logs/medsam_eval_%j.err

# ============================================================
# MedSAM Evaluation on BCSS Dataset (HPRC Grace)
# ============================================================
# MedSAM (Nature Communications 2024) - Medical Image Segmentation
# Trained on 1.57M medical image-mask pairs across 10 modalities
# 
# Note: MedSAM uses SAM ViT-B (86M params), smaller than SAM2 Hiera-L
# Local test showed: MedSAM 0.522 vs SAM2 0.553 (MedSAM worse!)
# ============================================================

echo "============================================================"
echo "SLURM Job: MedSAM Evaluation on HPRC Grace"
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

# Load modules for Grace cluster
module purge
module load GCC/11.3.0
module load CUDA/11.7.0
module load Python/3.10.4
module load cuDNN/8.4.1.50-CUDA-11.7.0

# Activate conda environment
source activate vfm || conda activate vfm || { echo "ERROR: Failed to activate vfm environment"; exit 1; }

# Check MedSAM is set up
if [ ! -d "MedSAM" ]; then
    echo "ERROR: MedSAM not found. Run setup first:"
    echo "  sbatch scripts/slurm/setup_medsam_hprc.slurm"
    exit 1
fi

# Check checkpoint exists
CHECKPOINT="models/medsam_checkpoints/medsam_vit_b.pth"
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: MedSAM checkpoint not found at $CHECKPOINT"
    echo "Run setup first: sbatch scripts/slurm/setup_medsam_hprc.slurm"
    exit 1
fi

echo ""
echo "Configuration:"
echo "  Checkpoint: $CHECKPOINT"
echo "  Dataset: BCSS test split (45 samples)"
echo ""

# ============================================================
# Experiment 1: MedSAM Box Prompt (baseline)
# ============================================================
echo ">>> Experiment 1: MedSAM Box Prompt (baseline)"
echo "    Expected time: ~3-5 minutes on A100"
python src/evaluate_medsam.py \
    --checkpoint "$CHECKPOINT" \
    --split test \
    --output_dir results/medsam_box_baseline \
    --tqdm

echo ""
echo "Experiment 1 Result:"
cat results/medsam_box_baseline/metrics.json | python -c "import sys,json; d=json.load(sys.stdin); print(f\"  Overall Dice: {d['overall']:.4f}\")" 2>/dev/null || echo "  (metrics not available)"

# ============================================================
# Experiment 2: MedSAM Box Prompt + TTA
# ============================================================
echo ""
echo ">>> Experiment 2: MedSAM Box Prompt + TTA"
echo "    Expected time: ~10-15 minutes on A100 (4x augmentations)"
python src/evaluate_medsam.py \
    --checkpoint "$CHECKPOINT" \
    --split test \
    --use_tta \
    --output_dir results/medsam_box_tta \
    --tqdm

echo ""
echo "Experiment 2 Result:"
cat results/medsam_box_tta/metrics.json | python -c "import sys,json; d=json.load(sys.stdin); print(f\"  Overall Dice: {d['overall']:.4f}\")" 2>/dev/null || echo "  (metrics not available)"

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================================"
echo "MedSAM Evaluation Complete!"
echo "Finished: $(date)"
echo "============================================================"

echo ""
echo "Results Summary:"
echo "================"

# Read and display results
if [ -f "results/medsam_box_baseline/metrics.json" ]; then
    echo ""
    echo "MedSAM Box Baseline:"
    python -c "
import json
with open('results/medsam_box_baseline/metrics.json') as f:
    d = json.load(f)
print(f\"  Tumor:        {d['tumor']['dice']:.4f}\")
print(f\"  Stroma:       {d['stroma']['dice']:.4f}\")
print(f\"  Lymphocyte:   {d['lymphocyte']['dice']:.4f}\")
print(f\"  Necrosis:     {d['necrosis']['dice']:.4f}\")
print(f\"  Blood Vessel: {d['blood_vessel']['dice']:.4f}\")
print(f\"  OVERALL:      {d['overall']:.4f}\")
"
fi

if [ -f "results/medsam_box_tta/metrics.json" ]; then
    echo ""
    echo "MedSAM Box + TTA:"
    python -c "
import json
with open('results/medsam_box_tta/metrics.json') as f:
    d = json.load(f)
print(f\"  Tumor:        {d['tumor']['dice']:.4f}\")
print(f\"  Stroma:       {d['stroma']['dice']:.4f}\")
print(f\"  Lymphocyte:   {d['lymphocyte']['dice']:.4f}\")
print(f\"  Necrosis:     {d['necrosis']['dice']:.4f}\")
print(f\"  Blood Vessel: {d['blood_vessel']['dice']:.4f}\")
print(f\"  OVERALL:      {d['overall']:.4f}\")
"
fi

echo ""
echo "Comparison with SAM2 (from local tests):"
echo "  SAM2 Box+Neg+TTA: 0.566 (BEST)"
echo "  SAM2 Box baseline: 0.553"
echo "  MedSAM Box baseline: 0.522 (7.8% worse than SAM2)"
echo ""
echo "Results saved to:"
echo "  results/medsam_box_baseline/metrics.json"
echo "  results/medsam_box_tta/metrics.json"
