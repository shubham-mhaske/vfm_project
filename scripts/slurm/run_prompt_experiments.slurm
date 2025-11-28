#!/bin/bash
#SBATCH --job-name=prompt_exp
#SBATCH --output=slurm_logs/prompt_exp_%j.out
#SBATCH --error=slurm_logs/prompt_exp_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=2:00:00

# ============================================================================
# Prompt Engineering Experiments for SAM2 on BCSS
# 
# Tests 6 configurations:
# 1. Box prompt (baseline)
# 2. Centroid (single point) prompt
# 3. Multi-point prompt
# 4. Box + negative points
# 5. Box + TTA
# 6. Box + negative points + TTA
# ============================================================================

echo "========================================"
echo "SAM2 Prompt Engineering Experiments"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started: $(date)"
echo "========================================"

cd /scratch/user/shubhammhaske/vfm_project
source ~/.bashrc
conda activate vfm

mkdir -p slurm_logs

# Verify GPU
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Run the experiments
bash scripts/run_prompt_experiments.sh

echo ""
echo "========================================"
echo "Completed: $(date)"
echo "========================================"
