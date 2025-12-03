#!/bin/bash
# Run all remaining experiments in parallel
# Execute from HPRC login node

echo "============================================"
echo "SUBMITTING ALL REMAINING EXPERIMENTS"
echo "============================================"

cd $SCRATCH/vfm_project
git pull

echo ""
echo "Submitting jobs..."

# 1. Ensemble CLIP + PLIP
JOB1=$(sbatch --parsable scripts/slurm/run_ensemble.slurm)
echo "  [1] Ensemble CLIP+PLIP: Job $JOB1"

# 2. Multi-scale CLIP
JOB2=$(sbatch --parsable scripts/slurm/run_multiscale.slurm)
echo "  [2] Multi-scale CLIP: Job $JOB2"

# 3. Pathology encoders (UNI/CTransPath)
JOB3=$(sbatch --parsable scripts/slurm/run_pathology_encoders.slurm)
echo "  [3] Pathology Encoders: Job $JOB3"

echo ""
echo "============================================"
echo "All jobs submitted!"
echo "============================================"
echo ""
echo "Monitor with: squeue -u \$USER"
echo ""
echo "Expected results:"
echo "  1. results/ensemble_clip_plip/    (~15 min)"
echo "  2. results/multiscale_clip/       (~15 min)"  
echo "  3. results/pathology_encoders/    (~30 min)"
echo ""
echo "View logs with:"
echo "  tail -f slurm_logs/ensemble_*.out"
echo "  tail -f slurm_logs/multiscale_*.out"
echo "  tail -f slurm_logs/path_encoders_*.out"
