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

### v4: Path-SAM2 + CTransPath (Baseline)
**Date**: 2025-11-26
**Config**: `path_sam2_ctranspath`
**Architecture**: SAM2 + CTransPath with attention fusion
**Result**: In progress. Expected Dice ~0.55-0.60

---

### v5: Path-SAM2 + CTransPath Optimized ✨ NEW
**Date**: 2025-11-27
**Config**: `path_sam2_ctranspath_optimized`
**Changes**:
- AMP bfloat16 enabled
- CTransPath frozen, `concat` fusion
- Batch size 6, 12 workers
- LR 6e-5, 40 epochs
**Results (Expected)**:

| Class        | Baseline | v5 Expected | Improvement |
|--------------|----------|-------------|-------------|
| Tumor        | 0.54     | 0.60-0.62   | +11-15%     |
| Blood Vessel | 0.03     | 0.10-0.15   | +233-400%   |
| **Overall**  | **0.42** | **0.55-0.60** | **+31-43%** |

---

### v6: + Focal Loss ✨ NEW
**Date**: 2025-11-27
**Config**: `path_sam2_focal`
**Changes**:
- Implemented `FocalDiceLoss` to address class imbalance.
- Class weights applied, especially for `blood_vessel`.
**Results (Expected)**:
- Overall Dice: 0.55-0.60 → **0.65-0.70**
- Blood Vessel Dice: 0.10-0.15 → **0.25-0.30**

---

### v7: + Stain Augmentation ✨ NEW
**Date**: 2025-11-27
**Config**: `path_sam2_focal_stain`
**Changes**:
- Added Macenko stain normalization and augmentation.
- Applied directly in the dataset loader.
**Results (Expected)**:
- Overall Dice: 0.65-0.70 → **0.68-0.72**
- Improved generalization across different medical centers.

---

## Commands

```bash
# Train a specific experiment
sbatch scripts/slurm/run_path_sam2_ctranspath_optimized.slurm

# Run the full evaluation pipeline for an experiment
bash scripts/evaluation/evaluate_experiment.sh finetune_logs/path_sam2_focal-2025-11-27_12-00-00

# Find optimal thresholds for a checkpoint
bash scripts/validation/run_threshold_search.sh <path_to_checkpoint.pt>

# Run per-class validation with TTA and optimal thresholds
python scripts/validation/validate_perclass.py \
  --checkpoint <path_to_checkpoint.pt> \
  --tta \
  --threshold_config configs/optimal_thresholds.json
```

---

**Last Updated**: November 27, 2025