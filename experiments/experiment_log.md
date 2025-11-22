# Experiment Log

---

## v1: base_finetune_v1

**Date**: 2025-11-21  
**Status**: ⚠️ Unstable - Collapsed at epoch 50

**Config**: LR 2e-5, cosine→1e-6, no warmup, loss weights 20:1:1:1, WD 0.05, ColorJitter 0.3, 150 epochs

**Results** (val, 21 samples):
| Epoch | Dice | IoU |
|-------|------|-----|
| 50 | 0.0716 | 0.0431 | 🔴 Collapse
| 80 | 0.3208 | 0.2067 | Recovered
| 100 | **0.3354** | **0.2232** | Best
| 110 | 0.2662 | 0.1721 | Degraded

**Issues**: Collapse at epoch 50, oscillating performance, low absolute scores

---

## v2: base_finetune_v2_stable

**Date**: TBD  
**Status**: ✅ Ready to launch

**Changes from v1**:
- LR: 2e-5→**7e-5**, warmup **125 steps**, end 1e-6→**7e-6**
- Batch: 4→**8** (40% faster, ~10h vs 16h)
- Loss: mask 20→**5**, dice 1→**2**
- Weight decay: 0.05→**0.01**
- ColorJitter: 0.3→**0.2**
- Epochs: 150→**100**, save freq: 10→**5**

**Target**: Dice >0.40, no collapse, stable training

**Launch**: `sbatch experiments/base_finetune_v2_stable/slurm_job.sh`

---

## Future Ideas

- LR sweep (find optimal)
- Smaller model (Hiera B+)
- Validation loop during training
- Progressive augmentation
