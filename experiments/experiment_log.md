# Experiment Log

Quick reference for experiment results. See `../EXPERIMENTS.md` for full details.

## 🚨 Key Finding

**Zeroshot SAM2 (0.517 Dice) outperforms ALL finetuned models.**

## Summary Table

| Version | Date | Overall Dice | vs Zeroshot | Status |
|---------|------|--------------|-------------|--------|
| **Zeroshot** | 2025-11-25 | **0.517** | baseline | ✅ Best |
| v2 | 2025-11-22 | 0.42 | -19% | ✅ Best Finetuned |
| v3 | 2025-11-26 | 0.40 | -22% | ⚠️ Worse |
| Path-SAM2+CTP | 2025-11-27 | 0.366 | -29% | ❌ Failed |
| LoRA-Light | 2025-11-27 | 0.355 | -31% | ❌ Failed |
| Box+Focal | 2025-11-27 | 0.372 | -28% | ❌ Failed |
| v1 | 2025-11-21 | 0.07 | -86% | ❌ Collapsed |

## Per-Class Results (Best Finetuned - v2)

| Class | Dice |
|-------|------|
| Tumor | 0.54 |
| Stroma | 0.43 |
| Necrosis | 0.45 |
| Lymphocyte | 0.20 |
| Blood Vessel | 0.03 |

## Per-Class Results (Box+Focal - Best for minorities)

| Class | Dice | vs v2 |
|-------|------|-------|
| Tumor | 0.386 | -29% |
| Stroma | 0.386 | -10% |
| Necrosis | 0.478 | +6% |
| Lymphocyte | 0.301 | +50% |
| Blood Vessel | 0.335 | **+1017%** |

## Key Insights

1. **Blood vessel**: Dramatically improved with class weighting (0.03 → 0.335)
2. **Majority classes**: Hurt by finetuning (tumor: 0.54 → 0.32-0.39)
3. **Trade-off**: Can't improve minorities without hurting majorities via finetuning
4. **Solution**: Use LoRA adapters (src/lora_adapter.py) - adds only 0.2% trainable params

## Next Steps

1. ✨ **LoRA Adapters** - Freeze original weights, add small trainable matrices
2. **TTA** - Test-time augmentation without any training
3. **Prompt engineering** - Better prompts for zeroshot

## Bug Fixes (2025-11-26)

- ✅ Fixed class mapping: blood_vessel is class 18, not 5
- ✅ Added training filter: only target classes {1,2,3,4,18}
- ✅ Fixed Path-SAM2 checkpoint loading

---

**Last Updated**: November 27, 2025
