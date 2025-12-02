#!/bin/bash
# Download CLIP model for offline use on HPRC compute nodes
# Run this on the LOGIN NODE (which has internet access)

set -e

CLIP_DIR="${SCRATCH}/clip_model"
echo "Downloading CLIP model to: $CLIP_DIR"

mkdir -p "$CLIP_DIR"
cd "$CLIP_DIR"

BASE_URL="https://huggingface.co/openai/clip-vit-base-patch32/resolve/main"

echo "Downloading model files..."

# Download each file with progress
wget -c "${BASE_URL}/config.json" -O config.json
echo "  ✓ config.json"

wget -c "${BASE_URL}/pytorch_model.bin" -O pytorch_model.bin
echo "  ✓ pytorch_model.bin (this is the large one ~350MB)"

wget -c "${BASE_URL}/preprocessor_config.json" -O preprocessor_config.json
echo "  ✓ preprocessor_config.json"

wget -c "${BASE_URL}/tokenizer.json" -O tokenizer.json
echo "  ✓ tokenizer.json"

wget -c "${BASE_URL}/tokenizer_config.json" -O tokenizer_config.json
echo "  ✓ tokenizer_config.json"

wget -c "${BASE_URL}/vocab.json" -O vocab.json
echo "  ✓ vocab.json"

wget -c "${BASE_URL}/merges.txt" -O merges.txt
echo "  ✓ merges.txt"

wget -c "${BASE_URL}/special_tokens_map.json" -O special_tokens_map.json
echo "  ✓ special_tokens_map.json"

echo ""
echo "Download complete! Files:"
ls -lh "$CLIP_DIR"

echo ""
echo "Total size: $(du -sh $CLIP_DIR | cut -f1)"
echo ""
echo "CLIP model is ready for offline use."
echo "The SLURM script will automatically detect it at: $CLIP_DIR"
