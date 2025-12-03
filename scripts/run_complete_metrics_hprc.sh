#!/bin/bash
#=============================================================
# Run Complete Metrics Collection on HPRC
#=============================================================
# This script submits the comprehensive metrics collection job
# to the HPRC Grace cluster.
#
# Usage:
#   cd /path/to/vfm_project
#   bash scripts/run_complete_metrics_hprc.sh
#
# Or submit directly:
#   sbatch scripts/slurm/run_complete_metrics.slurm
#=============================================================

echo "============================================"
echo "VFM Project - Complete Metrics Collection"
echo "============================================"
echo ""

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "ERROR: Please run this script from the project root directory"
    echo "       cd /path/to/vfm_project"
    echo "       bash scripts/run_complete_metrics_hprc.sh"
    exit 1
fi

# Create necessary directories
mkdir -p slurm_logs
mkdir -p results/complete_metrics

echo "Checking prerequisites..."

# Check for SAM2 checkpoint
if [ ! -f "sam2/checkpoints/sam2.1_hiera_large.pt" ]; then
    echo "WARNING: SAM2 checkpoint not found!"
    echo "         Run: bash sam2/checkpoints/download_ckpts.sh"
fi

# Check for MedSAM checkpoint
if [ ! -f "models/medsam_checkpoints/medsam_vit_b.pth" ]; then
    echo "WARNING: MedSAM checkpoint not found!"
    echo "         Run: bash scripts/download_medsam.sh"
    echo "         (MedSAM evaluation will be skipped)"
fi

# Check for BCSS data
if [ ! -d "data/bcss/images" ]; then
    echo "ERROR: BCSS dataset not found at data/bcss/"
    echo "       Please download/setup the dataset first"
    exit 1
fi

echo ""
echo "Submitting SLURM job..."
echo ""

# Submit the job
JOB_ID=$(sbatch scripts/slurm/run_complete_metrics.slurm | awk '{print $4}')

if [ -z "$JOB_ID" ]; then
    echo "ERROR: Failed to submit job"
    exit 1
fi

echo "============================================"
echo "Job submitted successfully!"
echo "============================================"
echo ""
echo "Job ID:        $JOB_ID"
echo "Output file:   slurm_logs/complete_metrics-$JOB_ID.out"
echo "Error file:    slurm_logs/complete_metrics-$JOB_ID.err"
echo "Results dir:   results/complete_metrics/"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f slurm_logs/complete_metrics-$JOB_ID.out"
echo ""
echo "Expected runtime: 4-6 hours"
echo "============================================"
