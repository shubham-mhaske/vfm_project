#!/bin/bash
#SBATCH --job-name=refinement_exp
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
#SBATCH --output=slurm_logs/refinement_exp_%j.out
#SBATCH --error=slurm_logs/refinement_exp_%j.err

# ============================================================
# Iterative Refinement Experiments for SAM2
# ============================================================
# Tests whether feeding predicted masks back to SAM2 improves
# segmentation quality. Expected improvement: +1-2% Dice.
# ============================================================

echo "============================================================"
echo "SLURM Job: Iterative Refinement Experiments"
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
echo "Running iterative refinement experiments..."
echo "Testing: baseline, refine2, refine3, TTA, TTA+refine"
echo ""

python src/evaluate_refinement.py \
    --model_cfg sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml \
    --checkpoint sam2/checkpoints/sam2.1_hiera_large.pt \
    --split test \
    --output_dir results/refinement_experiments

echo ""
echo "============================================================"
echo "Completed: $(date)"
echo "============================================================"

# Display results summary
if [ -f "results/refinement_experiments/refinement_results.json" ]; then
    echo ""
    echo "Results:"
    python -c "
import json
with open('results/refinement_experiments/refinement_results.json') as f:
    results = json.load(f)
baseline = results['box_neg_baseline']['overall']
print(f\"{'Config':<25} {'Dice':>10} {'vs Baseline':>12}\")
print('-'*50)
for name, r in results.items():
    dice = r['overall']
    diff = (dice - baseline) / baseline * 100
    diff_str = f'+{diff:.1f}%' if diff >= 0 else f'{diff:.1f}%'
    print(f'{name:<25} {dice:>10.4f} {diff_str:>12}')
"
fi
