#!/bin/bash
# Quick local test of the metrics collection script
# Run with: bash scripts/test_metrics_collection.sh

echo "============================================"
echo "Testing Metrics Collection (2 samples)"
echo "============================================"

cd "$(dirname "$0")/.."

# Test with just 2 samples to verify everything works
python scripts/collect_all_metrics.py \
    --local \
    --max_samples 2 \
    --skip_medsam

echo ""
echo "Test complete! Check results/complete_metrics/ for output."
