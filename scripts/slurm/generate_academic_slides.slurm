#!/bin/bash
#SBATCH --job-name=academic_slides
#SBATCH --output=slurm_logs/academic_slides-%j.out
#SBATCH --error=slurm_logs/academic_slides-%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=short

# Load modules
module purge
module load GCC/12.2.0
module load Python/3.10.8

# Activate virtual environment
source ~/vfm_env/bin/activate

# Change to project directory
cd $SLURM_SUBMIT_DIR

# Create output directories
mkdir -p results/figures/academic
mkdir -p presentation_data
mkdir -p slurm_logs

echo "=============================================="
echo "Step 1: Generating academic figures..."
echo "=============================================="
python scripts/analysis/generate_academic_slides.py

echo ""
echo "=============================================="
echo "Step 2: Generating full presentation..."
echo "=============================================="
python scripts/analysis/create_detailed_presentation.py

echo ""
echo "=============================================="
echo "Done! Output files:"
echo "=============================================="
echo "Academic figures:"
ls -la results/figures/academic/
echo ""
echo "Presentations:"
ls -la presentation_data/*.pptx

