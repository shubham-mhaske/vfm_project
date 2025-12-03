#!/bin/bash
# Download MedSAM checkpoint and clone repository
# Usage: bash scripts/download_medsam.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "==================================================="
echo "Downloading MedSAM for Medical Image Segmentation"
echo "==================================================="
echo ""

# Create directories
MEDSAM_DIR="$PROJECT_ROOT/MedSAM"
CHECKPOINT_DIR="$PROJECT_ROOT/models/medsam_checkpoints"
mkdir -p "$CHECKPOINT_DIR"

# Clone MedSAM repository if not exists
if [ ! -d "$MEDSAM_DIR" ]; then
    echo "[1/3] Cloning MedSAM repository..."
    git clone https://github.com/bowang-lab/MedSAM.git "$MEDSAM_DIR"
    echo "✓ MedSAM repository cloned to: $MEDSAM_DIR"
else
    echo "[1/3] MedSAM repository already exists at: $MEDSAM_DIR"
fi

# Check if checkpoint already exists
CHECKPOINT_PATH="$CHECKPOINT_DIR/medsam_vit_b.pth"
if [ -f "$CHECKPOINT_PATH" ]; then
    echo "[2/3] MedSAM checkpoint already exists at: $CHECKPOINT_PATH"
else
    echo "[2/3] Downloading MedSAM checkpoint..."
    echo ""
    echo "The checkpoint is hosted on Google Drive. You have two options:"
    echo ""
    echo "OPTION A: Download manually"
    echo "  1. Visit: https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN"
    echo "  2. Download 'medsam_vit_b.pth' (~375 MB)"
    echo "  3. Move it to: $CHECKPOINT_PATH"
    echo ""
    echo "OPTION B: Use gdown (if installed)"
    echo "  pip install gdown"
    echo "  gdown --id 1UAmWL88roYR7wKlnApw5Bcuzf2iQgk6_ -O $CHECKPOINT_PATH"
    echo ""
    
    # Try using gdown if available
    if command -v gdown &> /dev/null; then
        echo "gdown found! Attempting automatic download..."
        # MedSAM checkpoint file ID from Google Drive
        gdown --id 1UAmWL88roYR7wKlnApw5Bcuzf2iQgk6_ -O "$CHECKPOINT_PATH"
        if [ -f "$CHECKPOINT_PATH" ]; then
            echo "✓ MedSAM checkpoint downloaded successfully!"
        else
            echo "✗ Download failed. Please download manually."
            exit 1
        fi
    else
        echo "gdown not found. Installing gdown..."
        pip install gdown
        echo "Attempting download with gdown..."
        gdown --id 1UAmWL88roYR7wKlnApw5Bcuzf2iQgk6_ -O "$CHECKPOINT_PATH"
        if [ -f "$CHECKPOINT_PATH" ]; then
            echo "✓ MedSAM checkpoint downloaded successfully!"
        else
            echo "✗ Download failed. Please download manually from the link above."
            exit 1
        fi
    fi
fi

# Install MedSAM as a package
echo "[3/3] Installing MedSAM package..."
cd "$MEDSAM_DIR"
pip install -e . --quiet
echo "✓ MedSAM installed as editable package"

echo ""
echo "==================================================="
echo "MedSAM Setup Complete!"
echo "==================================================="
echo ""
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Repository: $MEDSAM_DIR"
echo ""
echo "To run evaluation:"
echo "  python src/evaluate_medsam.py --checkpoint models/medsam_checkpoints/medsam_vit_b.pth"
echo ""
