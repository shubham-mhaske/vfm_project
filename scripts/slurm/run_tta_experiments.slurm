#!/bin/bash
#SBATCH --job-name=tta_eval
#SBATCH --output=slurm_logs/tta_eval_%j.out
#SBATCH --error=slurm_logs/tta_eval_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=1:00:00

# ============================================================================
# TTA (Test-Time Augmentation) Experiments for SAM2
# ============================================================================

echo "========================================"
echo "SAM2 TTA Experiments"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started: $(date)"
echo "========================================"

cd /scratch/user/shubhammhaske/vfm_project
source ~/.bashrc
conda activate vfm

mkdir -p slurm_logs
mkdir -p results/prompt_experiments_20251127_212048

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

COMMON="--model_cfg configs/sam2.1/sam2.1_hiera_l.yaml --checkpoint sam2/checkpoints/sam2.1_hiera_large.pt --split test --tqdm"

echo ""
echo "[1/2] Box + TTA..."
python src/evaluate_segmentation.py $COMMON \
    --prompt_type box \
    --use_tta \
    --output_dir results/prompt_experiments_20251127_212048/box_tta

echo ""
echo "[2/2] Box + Neg Points + TTA..."
python src/evaluate_segmentation.py $COMMON \
    --prompt_type box \
    --use_tta \
    --use_neg_points \
    --output_dir results/prompt_experiments_20251127_212048/box_neg_tta

echo ""
echo "========================================"
echo "Completed: $(date)"
echo "========================================"

# Print summary
echo ""
echo "RESULTS:"
for dir in box_tta box_neg_tta; do
    path="results/prompt_experiments_20251127_212048/$dir/metrics.json"
    if [ -f "$path" ]; then
        dice=$(python -c "import json; d=json.load(open('$path')); print(f\"{d.get('overall', 'N/A'):.4f}\")" 2>/dev/null || echo "N/A")
        printf "%-20s : Dice = %s\n" "$dir" "$dice"
    fi
done
