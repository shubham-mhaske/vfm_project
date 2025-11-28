#!/bin/bash
#SBATCH --job-name=setup_medsam
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=short
#SBATCH --output=slurm_logs/setup_medsam_%j.out
#SBATCH --error=slurm_logs/setup_medsam_%j.err

# ============================================================
# Setup MedSAM on HPRC Grace Cluster
# ============================================================
# This script:
# 1. Clones MedSAM repository
# 2. Downloads MedSAM checkpoint (~358 MB)
# 3. Installs MedSAM package
# ============================================================

echo "============================================================"
echo "Setting up MedSAM on HPRC"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "============================================================"

# Navigate to project root
cd /scratch/user/$(whoami)/vfm_project || { echo "Project not found"; exit 1; }

# Create directories
mkdir -p slurm_logs
mkdir -p models/medsam_checkpoints

# Load modules
module purge
module load GCC/11.3.0
module load Python/3.10.4

# Activate conda environment
source activate vfm || conda activate vfm || { echo "Failed to activate vfm environment"; exit 1; }

# ============================================================
# Step 1: Clone MedSAM repository
# ============================================================
echo ""
echo "[1/3] Cloning MedSAM repository..."
if [ ! -d "MedSAM" ]; then
    git clone https://github.com/bowang-lab/MedSAM.git
    echo "✓ MedSAM cloned successfully"
else
    echo "✓ MedSAM already exists"
fi

# ============================================================
# Step 2: Download MedSAM checkpoint
# ============================================================
CHECKPOINT="models/medsam_checkpoints/medsam_vit_b.pth"
echo ""
echo "[2/3] Downloading MedSAM checkpoint..."

if [ -f "$CHECKPOINT" ]; then
    echo "✓ Checkpoint already exists: $CHECKPOINT"
else
    # Install gdown if not available
    pip install gdown --quiet
    
    # Download from Google Drive
    # File ID: 1UAmWL88roYR7wKlnApw5Bcuzf2iQgk6_
    echo "Downloading from Google Drive (~358 MB)..."
    gdown 1UAmWL88roYR7wKlnApw5Bcuzf2iQgk6_ -O "$CHECKPOINT"
    
    if [ -f "$CHECKPOINT" ]; then
        echo "✓ Checkpoint downloaded successfully"
        ls -lh "$CHECKPOINT"
    else
        echo "✗ Download failed!"
        echo "Please manually download from:"
        echo "  https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN"
        echo "And place at: $CHECKPOINT"
        exit 1
    fi
fi

# ============================================================
# Step 3: Install MedSAM package
# ============================================================
echo ""
echo "[3/3] Installing MedSAM package..."
cd MedSAM
pip install -e . --quiet
cd ..
echo "✓ MedSAM installed"

# Verify installation
echo ""
echo "============================================================"
echo "Verification"
echo "============================================================"
python -c "from segment_anything import sam_model_registry; print('✓ MedSAM import successful')"

echo ""
echo "============================================================"
echo "MedSAM Setup Complete!"
echo "Finished: $(date)"
echo "============================================================"
echo ""
echo "Next step: Submit evaluation job"
echo "  sbatch scripts/slurm/run_medsam_eval.slurm"
