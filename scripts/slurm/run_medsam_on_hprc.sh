#!/bin/bash
# ============================================================
# Sync and Run MedSAM Evaluation on HPRC Grace
# ============================================================
# Usage: bash scripts/run_medsam_on_hprc.sh
# ============================================================

set -e

# Configuration
HPRC_USER="${HPRC_USER:-$(whoami)}"
HPRC_HOST="grace.hprc.tamu.edu"
REMOTE_DIR="/scratch/user/${HPRC_USER}/vfm_project"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "============================================================"
echo "MedSAM Evaluation on HPRC Grace"
echo "============================================================"
echo "Local:  $LOCAL_DIR"
echo "Remote: ${HPRC_USER}@${HPRC_HOST}:${REMOTE_DIR}"
echo "============================================================"
echo ""

# Step 1: Sync project to HPRC
echo "[1/4] Syncing project to HPRC..."
rsync -avz --progress \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.DS_Store' \
    --exclude='finetune_logs/' \
    --exclude='slurm_logs/*.out' \
    --exclude='slurm_logs/*.err' \
    "$LOCAL_DIR/" "${HPRC_USER}@${HPRC_HOST}:${REMOTE_DIR}/"

echo ""
echo "[2/4] Setting up MedSAM on HPRC (if needed)..."
ssh "${HPRC_USER}@${HPRC_HOST}" << 'EOF'
cd /scratch/user/$(whoami)/vfm_project

# Check if MedSAM checkpoint exists
if [ ! -f "models/medsam_checkpoints/medsam_vit_b.pth" ]; then
    echo "MedSAM checkpoint not found. Submitting setup job..."
    sbatch scripts/slurm/setup_medsam_hprc.slurm
    echo "Setup job submitted. Wait for it to complete before running evaluation."
    echo "Check status: squeue -u $(whoami)"
    exit 0
fi

echo "✓ MedSAM checkpoint found"
EOF

echo ""
echo "[3/4] Submitting MedSAM evaluation job..."
ssh "${HPRC_USER}@${HPRC_HOST}" << 'EOF'
cd /scratch/user/$(whoami)/vfm_project

# Check checkpoint exists
if [ ! -f "models/medsam_checkpoints/medsam_vit_b.pth" ]; then
    echo "Waiting for setup job to complete..."
    exit 1
fi

# Submit evaluation job
JOB_ID=$(sbatch --parsable scripts/slurm/run_medsam_eval.slurm)
echo "Submitted job: $JOB_ID"
echo ""
echo "Monitor with:"
echo "  squeue -u $(whoami)"
echo "  tail -f slurm_logs/medsam_eval_${JOB_ID}.out"
EOF

echo ""
echo "[4/4] Done!"
echo ""
echo "To check results after job completes:"
echo "  ssh ${HPRC_USER}@${HPRC_HOST} 'cat ${REMOTE_DIR}/results/medsam_box_baseline/metrics.json'"
echo "  ssh ${HPRC_USER}@${HPRC_HOST} 'cat ${REMOTE_DIR}/results/medsam_box_tta/metrics.json'"
echo ""
echo "To sync results back:"
echo "  rsync -avz ${HPRC_USER}@${HPRC_HOST}:${REMOTE_DIR}/results/ ./results/"
