#!/bin/bash
# Submit all metrics collection jobs in parallel
# Run this from the project root on HPRC

cd /scratch/user/$(whoami)/vfm_project

# Create slurm_logs directory
mkdir -p slurm_logs

echo "Submitting all metrics collection jobs..."
echo ""

# Submit SAM2 job
SAM2_JOB=$(sbatch scripts/slurm/run_sam2_metrics.slurm | awk '{print $4}')
echo "SAM2 job submitted: $SAM2_JOB"

# Submit CLIP job
CLIP_JOB=$(sbatch scripts/slurm/run_clip_metrics.slurm | awk '{print $4}')
echo "CLIP job submitted: $CLIP_JOB"

# Submit MedSAM job
MEDSAM_JOB=$(sbatch scripts/slurm/run_medsam_metrics.slurm | awk '{print $4}')
echo "MedSAM job submitted: $MEDSAM_JOB"

echo ""
echo "All jobs submitted!"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo ""
echo "Check outputs:"
echo "  tail -f slurm_logs/sam2_metrics_${SAM2_JOB}.out"
echo "  tail -f slurm_logs/clip_metrics_${CLIP_JOB}.out"
echo "  tail -f slurm_logs/medsam_metrics_${MEDSAM_JOB}.out"
echo ""
echo "Results will be saved to: results/complete_metrics/"
