# base_finetune_v1 Results

**Date**: 2025-11-21  
**Run**: `finetune_logs/base_finetune-2025-11-21_20-45-29/`

## Config
- Model: SAM2.1 Hiera Large
- Dataset: BCSS (85 train / 21 val / 45 test)
- LR: 2e-5, cosine→1e-6, no warmup
- Loss: mask:20, dice:1, iou:1, class:1
- Weight decay: 0.05
- ColorJitter: 0.3
- 150 epochs, batch 4, resolution 1024

## Validation Results

| Epoch | Dice | IoU | Status |
|-------|------|-----|--------|
| 50 | 0.0716 | 0.0431 | 🔴 Collapsed |
| 80 | 0.3208 | 0.2067 | Recovered |
| 90 | 0.2870 | 0.1830 | Degraded |
| 100 | **0.3354** | **0.2232** | ✅ Best |
| 110 | 0.2662 | 0.1721 | Degraded |

## Issues
1. Collapse at epoch 50 (Dice→0.07)
2. Oscillating performance after recovery
3. Low absolute scores (target: Dice >0.40)

## Root Causes
- No warmup → early instability
- Aggressive LR decay (2e-5→1e-6)
- Imbalanced loss (mask 20x dominant)
- High weight decay (0.05) for small dataset

## v2 Fixes
- Add warmup (250 steps)
- Increase LR (5e-5), less decay (→5e-6)
- Balance loss (mask:5, dice:2)
- Reduce weight decay (0.01)
- Milder ColorJitter (0.2)
