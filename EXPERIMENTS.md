# Experiment Tracker: SAM2 Histopathology Segmentation

## 🚨 Critical Finding: Finetuning HURTS Zeroshot Performance

**All finetuning approaches tested so far have REDUCED performance compared to zeroshot SAM2.**

| Approach | Overall Dice | vs Zeroshot |
|----------|--------------|-------------|
| **Zeroshot SAM2** | **0.517** | **baseline** |
| v2 Finetuned (best finetune) | 0.42 | -19% |
| Path-SAM2 + CTransPath | 0.366 | -29% |
| SAM2 LoRA-Light | 0.355 | -31% |
| SAM2 Box+Focal (ep15) | 0.372 | -28% |

**Root Cause Analysis:**
1. **Dataset size**: Only 85 training images vs SAM's billions of training samples
2. **Catastrophic forgetting**: Full/partial finetuning destroys pretrained knowledge
3. **Domain mismatch**: Histopathology has very different visual patterns than natural images

**Recommended Next Steps:**
1. ✨ **LoRA Adapters** (src/lora_adapter.py) - Add ~0.2% trainable params, keep original frozen
2. **Test-Time Adaptation (TTA)** - Augment at inference, no training
3. **Prompt Engineering** - Better prompts may outperform finetuning

---

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

## Results Summary (All Experiments)

### Per-Class Dice Scores

| Experiment | Tumor | Stroma | Lymph | Necrosis | Blood V. | **Overall** |
|------------|-------|--------|-------|----------|----------|-------------|
| **Zeroshot SAM2** | - | - | - | - | - | **0.517** |
| v2 Finetuned | 0.54 | 0.43 | 0.20 | 0.45 | 0.03 | 0.42 |
| Path-SAM2+CTP | 0.345 | 0.388 | 0.330 | 0.530 | 0.283 | 0.366 |
| LoRA-Light | 0.320 | 0.394 | 0.265 | 0.498 | 0.350 | 0.355 |
| Box+Focal (ep15) | 0.386 | 0.386 | 0.301 | 0.478 | 0.335 | 0.372 |

**Key Insight**: Blood vessel (0.5% of data) improves most with specialized training (0.03 → 0.35), but at the cost of majority class performance.

---

## Detailed Training History

### Zeroshot SAM2 ✅ BEST OVERALL
**Date**: 2025-11-25  
**Config**: No training, uses pretrained SAM2 Hiera-L with CLIP classification
**Result**: **Dice 0.517, IoU 0.374**
**Note**: This is our strongest baseline - all finetuning has failed to improve it.

---

### v1: base_finetune_v1 ❌ FAILED
**Date**: 2025-11-21  
**Config**: LR 2e-5, no warmup, loss 20:1:1:1, 150 epochs  
**Result**: Collapsed at epoch 50 (Dice 0.07), unstable training

---

### v2: base_finetune_v2_stable ✅ BEST FINETUNED
**Date**: 2025-11-22  
**Config**:
- LR: 6e-5 with 175-step warmup → 6e-6 cosine decay
- Batch: 6, frozen encoder, 100 epochs
- Loss: mask 5, dice 2, iou 1, class 1
- Augmentation: ColorJitter 0.2, rotation 90°

**Results (Epoch 15, Test Set)**:

| Class | Dice |
|-------|------|
| Tumor | 0.54 |
| Stroma | 0.43 |
| Lymphocyte | 0.20 |
| Necrosis | 0.45 |
| Blood Vessel | 0.03 |
| **Overall** | **0.42** |

**Analysis**: Better than v1, but still 19% worse than zeroshot.

---

### v3: base_finetune_v3_perclass ⚠️ WORSE
**Date**: 2025-11-26  
**Config**: LR 8e-5, ColorJitter 0.3, warmup 250 steps, 40 epochs  
**Result**: Overall Dice 0.404 (-4% vs v2, -22% vs zeroshot)  
**Analysis**: Stronger augmentation too aggressive for H&E features

---

### Path-SAM2 + CTransPath ❌ WORSE THAN ZEROSHOT
**Date**: 2025-11-27  
**Config**: SAM2 + CTransPath pathology encoder with attention fusion  
**Results**:

| Class | Dice | Std |
|-------|------|-----|
| Tumor | 0.345 | 0.215 |
| Stroma | 0.388 | 0.246 |
| Lymphocyte | 0.330 | 0.262 |
| Necrosis | 0.530 | 0.309 |
| Blood Vessel | 0.283 | 0.241 |
| **Overall** | **0.366** | - |

**Analysis**: CTransPath features didn't help; may be incompatible with SAM2 architecture. 29% worse than zeroshot.

---

### SAM2 LoRA-Light ❌ WORSE THAN ZEROSHOT
**Date**: 2025-11-27  
**Config**: sam2_lora_light
- LR: 5e-6 (12× smaller than v2)
- Epochs: 30
- Warmup: 25%
- Minimal augmentation

**Results**:

| Class | Dice | Std |
|-------|------|-----|
| Tumor | 0.320 | 0.217 |
| Stroma | 0.394 | 0.229 |
| Lymphocyte | 0.265 | 0.229 |
| Necrosis | 0.498 | 0.311 |
| Blood Vessel | 0.350 | 0.233 |
| **Overall** | **0.355** | - |

**Analysis**: Even very conservative finetuning hurts. Blood vessel improved but overall 31% worse than zeroshot.

---

### SAM2 Box+Focal ❌ WORSE THAN ZEROSHOT
**Date**: 2025-11-27  
**Config**: sam2_box_focal
- Box-only prompts
- FocalDiceLoss with class weights
- LR: 3e-5, 50 epochs

**Results (Epoch 15)**:

| Class | Dice | Std |
|-------|------|-----|
| Tumor | 0.386 | 0.237 |
| Stroma | 0.386 | 0.231 |
| Lymphocyte | 0.301 | 0.251 |
| Necrosis | 0.478 | 0.298 |
| Blood Vessel | 0.335 | 0.240 |
| **Overall** | **0.372** | - |

**Analysis**: Best finetuned minority class performance, but still 28% worse than zeroshot overall.

---

## New Approaches to Try

### LoRA Adapters (Implemented: src/lora_adapter.py)

**Rationale**: Standard finetuning (even conservative) destroys SAM2's pretrained knowledge. LoRA adds ~0.2% new trainable parameters while keeping original weights frozen.

**Implementation**:
\`\`\`python
from src.lora_adapter import apply_lora_to_sam2

# Load SAM2
sam2_model = build_sam2(cfg_path, checkpoint_path)

# Apply LoRA (adds ~0.2% trainable params)
lora_model = apply_lora_to_sam2(
    sam2_model,
    r=8,              # rank (4, 8, or 16)
    target_modules='image_encoder',  # or 'all'
    trainable_output_head=True
)

# Train only LoRA params
optimizer = torch.optim.AdamW(lora_model.get_trainable_params(), lr=1e-4)
\`\`\`

**Key Design**:
- Freezes all 224M original SAM2 parameters
- Adds small LoRA matrices (A, B) to attention layers
- Output = Original + α/r × (B @ A @ x)
- Inspired by Medical SAM Adapter (arXiv:2304.12620)

---

### Test-Time Adaptation (TTA)

Apply augmentations at inference and aggregate predictions:
\`\`\`python
from src.tta_utils import apply_tta

predictions = apply_tta(model, image, augmentations=['hflip', 'vflip', 'rot90'])
\`\`\`

---

### Research Papers for Reference

1. **Medical SAM Adapter** (arXiv:2304.12620): Lightweight adapters for medical SAM
2. **SAM-Med2D** (arXiv:2308.16184): Fine-tuned on 4.6M medical images
3. **MedSAM** (Nature Communications): Fine-tuned on 1.57M medical images

**Key insight**: Successful medical SAM adaptations use 1.5M-4.6M images. With only 85 images, preserving pretrained knowledge is critical.

---

## Commands

\`\`\`bash
# --- Evaluate Zeroshot (Current Best) ---
python src/evaluation.py \\
  --sam_model_cfg configs/sam2.1/sam2.1_hiera_l.yaml \\
  --sam_checkpoint sam2/checkpoints/sam2.1_hiera_large.pt \\
  --output_dir results/zeroshot_eval

# --- Train with LoRA Adapters (New) ---
python src/train_with_lora.py \\
  --lora_rank 8 \\
  --target_modules image_encoder \\
  --lr 1e-4 \\
  --epochs 20

# --- Previous Experiments (for reference) ---
# These all performed worse than zeroshot
sbatch scripts/slurm/run_sam2_lora_light.slurm
sbatch scripts/slurm/run_sam2_box_focal.slurm

# --- Evaluation ---
python src/evaluate_segmentation.py \\
  --checkpoint path/to/checkpoint.pt \\
  --output_dir results/my_eval
\`\`\`

---

## Key Lessons Learned

1. **Zeroshot > Finetuning** for small datasets (~85 images)
2. **Catastrophic forgetting** is the main failure mode
3. **Class imbalance** (blood vessel 0.5%) benefits from weighted losses but hurts majority classes
4. **CTransPath integration** didn't help - architecture mismatch
5. **LoRA adapters** are the most promising path forward

---

**Last Updated**: November 27, 2025
