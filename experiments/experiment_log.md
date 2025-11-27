# Experiment Log

Quick reference for experiment results. See `../EXPERIMENTS.md` for full details.

## Summary Table

| Version | Date | Overall Dice | Status | Notes |
|---------|------|--------------|--------|-------|
| v1 | 2025-11-21 | 0.34 | ❌ Failed | Collapsed at epoch 50 |
| **v2** | 2025-11-22 | **0.42** | ✅ Best | Stable, use as baseline |
| v3 | 2025-11-26 | 0.40 | ⚠️ Worse | Aggressive augmentation hurt |

## Per-Class Results (v2, Epoch 15)

| Class | Dice |
|-------|------|
| Tumor | 0.54 |
| Stroma | 0.43 |
| Necrosis | 0.45 |
| Lymphocyte | 0.20 |
| Blood Vessel | 0.03 |

## Bug Fixes (2025-11-26)

- ✅ Fixed class mapping: blood_vessel is class 18, not 5
- ✅ Added training filter: only target classes {1,2,3,4,18}

**Recommendation**: Re-train with fixes for proper blood_vessel learning.

