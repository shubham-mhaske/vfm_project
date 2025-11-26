#!/bin/bash
# Quick local test of v3 config (1 epoch, 2 samples)

set -e

# Get project root (2 levels up from scripts/training/)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "============================================"
echo "Testing v3 Config Locally"
echo "============================================"
echo "Project root: $PROJECT_ROOT"
echo ""

# Test 1: Validate config syntax
echo -e "\n[1/3] Validating config syntax..."
python scripts/validate_config.py --config conf/experiment/base_finetune_v3_perclass.yaml

# Test 2: Dry run (0 epochs)
echo -e "\n[2/3] Dry run test..."
python src/train_sam.py \
  experiment=base_finetune_v3_perclass \
  scratch.num_epochs=0 \
  scratch.train_batch_size=2 \
  trainer.checkpoint.save_freq=1 \
  trainer.logging.log_freq=5 \
  +dry_run=true

# Test 3: Single epoch test
echo -e "\n[3/3] Single epoch test..."
python src/train_sam.py \
  experiment=base_finetune_v3_perclass \
  scratch.num_epochs=1 \
  scratch.train_batch_size=2 \
  trainer.checkpoint.save_freq=1 \
  trainer.logging.log_freq=5 \
  +test_run=true

echo -e "\n============================================"
echo "✅ v3 Config Test Complete!"
echo "============================================"
echo ""
echo "If all tests passed, you can run full training:"
echo "  sbatch run_training_grace.slurm base_finetune_v3_perclass"
echo ""
