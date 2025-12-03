#!/bin/bash
#SBATCH --job-name=qual_figures
#SBATCH --output=slurm_logs/qual_figures_%j.out
#SBATCH --error=slurm_logs/qual_figures_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu

# =============================================================================
# SLURM Job: Generate Qualitative Results Figures
# Runs SAM2 and MedSAM inference on test images and creates visualizations
# =============================================================================

echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=============================================="

# Load modules
module purge
module restore dl

# Navigate to project root
cd $SLURM_SUBMIT_DIR/..
PROJECT_ROOT=$(pwd)
echo "Project root: $PROJECT_ROOT"

# Activate virtual environment
source $SCRATCH/vfm_env/bin/activate
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
# Verify GPU
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo ""

# Check CUDA availability in Python
python -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Ensure output directory exists
mkdir -p results/figures/qualitative
mkdir -p slurm_logs

# Set environment variables for better GPU memory management
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Run the qualitative figure generation script
echo ""
echo "=============================================="
echo "Running Qualitative Figure Generation..."
echo "=============================================="

cd $PROJECT_ROOT
python scripts/analysis/generate_qualitative_results.py

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=============================================="
    echo "SUCCESS: Figures generated!"
    echo "=============================================="
    
    # List generated files
    echo "Generated files:"
    ls -la results/figures/qualitative/
    
    # Copy to presentation directory
    mkdir -p presentation_data/figures/qualitative
    cp -r results/figures/qualitative/* presentation_data/figures/qualitative/
    echo ""
    echo "Copied to presentation_data/figures/qualitative/"
else
    echo ""
    echo "=============================================="
    echo "ERROR: Figure generation failed!"
    echo "=============================================="
    exit 1
fi

echo ""
echo "End Time: $(date)"
echo "=============================================="
