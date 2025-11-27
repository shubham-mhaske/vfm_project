# Experiment Tracker: SAM2 Histopathology Segmentation

## Dataset Overview

**BCSS (Breast Cancer Semantic Segmentation)**
- **Samples**: 151 total (85 train, 21 val, 45 test)
- **Resolution**: 1024×1024 H&E stained tissue patches
- **Target Classes**: tumor (1), stroma (2), lymphocyte (3), necrosis (4), blood_vessel (18)

### Class Distribution

| Class | ID | Frequency | Weight Needed |
|-------|-----|-----------|---------------|
| Tumor | 1 | 33.8% | 1.0× (baseline) |
| Stroma | 2 | 26.2% | 1.3× |
| Necrosis | 4 | 6.9% | 4.9× |
| Lymphocyte | 3 | 5.9% | 5.7× |
| Blood Vessel | 18 | 0.5% | **67.6×** |

**⚠️ Critical**: Blood vessel is class ID **18**, not 5. Class 5 is glandular_secretions.

---

## Training History

### v1: base_finetune_v1 ❌ FAILED
**Date**: 2025-11-21  
**Config**: LR 2e-5, no warmup, loss 20:1:1:1, 150 epochs  
**Result**: Collapsed at epoch 50 (Dice 0.07), unstable training

---

### v2: base_finetune_v2_stable ✅ BEST
**Date**: 2025-11-22  
**Config**:
- LR: 6e-5 with 175-step warmup → 6e-6 cosine decay
- Batch: 6, frozen encoder, 100 epochs
- Loss: mask 5, dice 2, iou 1, class 1
- Augmentation: ColorJitter 0.2, rotation 90°

**Results (Epoch 15, Test Set)**:

| Class | Dice | IoU |
|-------|------|-----|
| Tumor | 0.54 | - |
| Stroma | 0.43 | - |
| Lymphocyte | 0.20 | - |
| Necrosis | 0.45 | - |
| Blood Vessel | 0.03 | - |
| **Overall** | **0.42** | **0.29** |

---

### v3: base_finetune_v3_perclass ⚠️ SLIGHTLY WORSE
**Date**: 2025-11-26  
**Config**: LR 8e-5, ColorJitter 0.3, warmup 250 steps, 40 epochs  
**Result**: Overall Dice 0.404 (-4% vs v2)  
**Analysis**: Stronger augmentation may be too aggressive for H&E features

---

## Bug Fixes Applied (2025-11-26)

### 1. Class Mapping Fix
- **Before**: `dataset.py` mapped class 5 → blood_vessel (WRONG)
- **After**: Correctly maps class 18 → blood_vessel
- **Impact**: Blood vessel evaluation was completely wrong

### 2. Training Class Filtering
- **Before**: Trained on all 17+ BCSS classes (~36% wasted effort)
- **After**: Filters to target classes only: {1, 2, 3, 4, 18}
- **Impact**: 100% training efficiency on relevant classes

---

## Next Steps

### v4: Re-train with Bug Fixes (Recommended)
1. Use corrected class mapping (class 18 = blood_vessel)
2. Train only on target classes {1, 2, 3, 4, 18}
3. Add class weighting for blood_vessel (67× weight)
4. Expected improvement: Blood vessel Dice 0.03 → 0.15+

### Future Experiments
- Gradual encoder unfreezing (epochs 20+)
- H&E stain normalization augmentation
- Multi-scale training (512→1024)
- Alternative architectures if SAM plateaus

---

## Commands

```bash
# Training
python src/run_finetuning.py experiment=base_finetune_v2_stable

# Evaluation
python src/evaluation.py \
  --sam_checkpoint finetune_logs/*/checkpoints/checkpoint_15.pt \
  --sam_model_cfg sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml \
  --clip_prompts configs/prompts/hard_coded_prompts_v2.json \
  --output_dir results/evaluation

# Per-class validation
python scripts/validation/validate_perclass.py \
  --checkpoint finetune_logs/*/checkpoints/checkpoint_15.pt \
  --split val
```

---

**Last Updated**: November 26, 2025
