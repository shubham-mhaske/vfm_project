# Experiment Tracker: SAM2 Histopathology Segmentation

## Dataset: BCSS (Breast Cancer Semantic Segmentation)
- **Task**: Multi-class tissue segmentation (tumor, stroma, lymphocyte, necrosis, blood_vessel)
- **Samples**: 151 total (85 train, 21 val, 45 test)
- **Resolution**: 1024×1024 patches from whole-slide images
- **Staining**: H&E (Hematoxylin and Eosin)
- **Class imbalance**: tumor 27%, stroma 23%, lymphocyte 7%, necrosis 8%, blood_vessel 2%

---

## Training History

### v1: base_finetune_v1 ❌ FAILED
**Date**: 2025-11-21  
**Config**: LR 2e-5, no warmup, loss 20:1:1:1, WD 0.05, ColorJitter 0.3, 150 epochs  
**Result**: Collapsed at epoch 50 (Dice 0.07), recovered to 0.34 by epoch 100  
**Issue**: Training instability, low performance

---

### v2: base_finetune_v2_stable ⚠️ MISLEADING VALIDATION
**Date**: 2025-11-22  
**Config**:
- LR: 6e-5 with 175-step warmup → 6e-6 cosine
- Batch: 6, frozen encoder, 100 epochs
- Loss: mask 5, dice 2, iou 1, class 1
- Augmentation: ColorJitter 0.2, rotation 90°

**Training Results** (validation during training):
| Epoch | Val Dice | Val IoU | Note |
|-------|----------|---------|------|
| 5 | 0.7843 | 0.6776 | Fast learning |
| 10 | 0.8138 | 0.7117 | Excellent |
| **15** | **0.8235** | **0.7226** | **Best** |
| 20 | 0.8178 | 0.7142 | Slight drop |
| 25 | 0.7813 | 0.6783 | Overfitting |

**CRITICAL DISCOVERY** (2025-11-25):
- Validation metric was **WRONG** - evaluated "any tissue vs background" (easy)
- True per-class evaluation: **Dice 0.42** (not 0.82!)
- Training learned tissue detection, NOT class discrimination

**Actual Test Performance** (per-class, epoch 15):
- Segmentation Dice: **0.4183**
- Segmentation IoU: **0.2860**
- CLIP Accuracy: **0.3013**
- Status: Much worse than training suggested

---

## Research Findings: Histopathology Segmentation

### Challenge: Why is performance low?
1. **Class similarity**: Tumor vs stroma have similar texture/color in H&E
2. **Boundary ambiguity**: Expert pathologists struggle with precise boundaries
3. **Stain variation**: H&E staining not standardized across slides
4. **SAM limitation**: Pretrained on natural images, not optimized for microscopy

### Literature Review: SOTA Approaches

**Medical Segmentation Models**:
- UNet++/TransUNet: Dice 0.75-0.85 on histopathology
- Mask2Former: Dice 0.78-0.82 on BCSS specifically
- SAM zero-shot on medical: Dice 0.30-0.55 (consistent with our finding)

**Key Insight**: SAM struggles with:
- Fine-grained histology features (needs finetuning)
- Per-class discrimination (needs class-aware training)
- Microscopy domain gap (needs domain adaptation)

### Dataset-Specific Challenges

**Class Imbalance** (from 20-sample analysis):
```
tumor:        27.13% ████████████████████████
stroma:       23.49% ████████████████████
lymphocyte:    7.05% ██████
necrosis:      7.76% ██████
blood_vessel:  1.71% █  ← Severely underrepresented!
```

**Implications**:
- Blood vessel class needs 15× more weight
- Lymphocyte needs 4× more weight
- Current training treats all classes equally → biased toward tumor/stroma

---

## Recommended Training Strategy: v3

### Core Improvements

**1. Fix Validation Metric** 🎯 CRITICAL
```python
# Current (WRONG): Evaluates "any tissue vs background"
binary_mask = (gt_mask > 0).astype(np.uint8)

# Correct: Evaluate per-class segmentation
for class_id in [1, 2, 3, 4, 9]:
    binary_mask = (gt_mask == class_id).astype(np.uint8)
    dice = evaluate_sam(image, binary_mask)
    class_dice[class_id].append(dice)
avg_dice = mean(class_dice.values())  # True metric!
```

**2. Class-Balanced Loss**
```yaml
loss:
  weight_dict:
    loss_mask: 5
    loss_dice: 2
    loss_iou: 1
    loss_class: 1
  class_weights:  # NEW: Weight by inverse frequency
    tumor: 1.0        # 27% → baseline
    stroma: 1.2       # 23% → slight boost
    lymphocyte: 3.9   # 7% → 4x weight
    necrosis: 3.5     # 8% → 3.5x weight
    blood_vessel: 15.9  # 2% → 15x weight!
```

**3. Gradual Encoder Unfreezing**
- Epochs 0-10: Frozen encoder (current approach)
- Epochs 11-25: Unfreeze stages 3-4 (last 2 layers), LR 6e-6
- Epochs 26-40: Unfreeze all, LR 6e-7
- Rationale: Encoder needs histology-specific features

**4. Stain Normalization** (Domain-Specific Aug)
```yaml
train_transforms:
  - RandomHEStainAugmentation:  # H&E specific
      alpha: 0.3
      beta: 0.3
  - RandomElasticDeformation:   # Tissue warping
      alpha: 30
      sigma: 4
  - GaussianBlur: {sigma: [0.1, 2.0], p: 0.3}
  - Existing augmentations...
```

**5. Training Schedule**
```yaml
scratch:
  num_epochs: 40  # Reduced from 100 (early stopping expected)
  base_lr: 8e-5   # Slightly higher for unfrozen encoder
  warmup_steps: 250  # Longer warmup for stability
  
optim:
  lr_schedule:
    - Linear warmup: 0 → 8e-5 (epochs 0-2, 20%)
    - Cosine decay: 8e-5 → 1e-6 (epochs 2-40, 80%)
  weight_decay: 0.02  # Higher for better generalization
  
  early_stopping:
    patience: 10
    monitor: val_perclass_dice  # NEW metric!
    min_delta: 0.001
```

**6. Multi-Scale Training** (Optional)
- Train on 512×512 for epochs 0-10 (faster)
- Switch to 1024×1024 for epochs 11+ (full resolution)
- Improves small class detection (blood vessels)

### Expected Improvements

| Metric | v2 (current) | v3 (target) | Improvement |
|--------|--------------|-------------|-------------|
| **Per-class Dice** | 0.42 | **0.60-0.70** | +43-67% |
| Per-class IoU | 0.29 | 0.45-0.55 | +55-90% |
| CLIP Accuracy | 0.30 | 0.50-0.60 | +67-100% |
| Training Dice | 0.82 (fake) | 0.60-0.70 (real) | Honest metric |

### Why These Improvements Work

1. **Class-balanced loss**: Forces model to learn rare classes (blood vessels)
2. **Gradual unfreezing**: Adapts encoder to histology textures
3. **Stain normalization**: Handles H&E variability across slides
4. **Correct validation**: Monitors actual task performance
5. **Multi-scale**: Captures both large structures (tumor) and small (vessels)

---

## Implementation Plan

### Phase 1: Local Testing (1 day)
```bash
# 1. Create v3 config
cp conf/experiment/base_finetune_v2_stable.yaml conf/experiment/base_finetune_v3_perclass.yaml
# Edit: add class weights, change validation, update schedule

# 2. Test class-balanced loss locally (1 epoch)
python src/test_training.py --config base_finetune_v3_perclass --max_epochs 1

# 3. Verify per-class validation works
python src/validate_perclass.py --checkpoint test_checkpoint.pt
```

### Phase 2: HPRC Training (4-5 hours)
```bash
# Upload code
scp -r conf/ src/ grace1:~/vfm_project/

# Submit job
ssh grace1 "cd vfm_project && sbatch run_training_grace.slurm base_finetune_v3_perclass"

# Monitor (watch for realistic Dice ~0.50-0.60, not 0.80+)
ssh grace1 "tail -f finetune_logs/base_finetune_v3*/logs/*.log | grep -E '(Dice|epoch)'"
```

### Phase 3: Evaluation (30 min)
```bash
# Download best checkpoint (expect epoch ~15-25)
scp grace1:~/vfm_project/finetune_logs/base_finetune_v3*/checkpoints/checkpoint_*.pt .

# Run per-class evaluation
python src/evaluation.py \
  --sam_checkpoint checkpoint_best.pt \
  --sam_model_cfg sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml \
  --clip_prompts configs/prompts/hard_coded_prompts_v2.json \
  --output_dir results/v3_evaluation

# Compare with v2
python experiments/compare_v2_v3.py
```

---

## Alternative: If v3 Doesn't Reach Dice 0.65+

### Plan B: Switch Architecture
SAM may not be optimal for histopathology. Consider:

1. **Mask2Former** (SOTA for medical segmentation)
   - Pre-trained on medical images
   - Better multi-class handling
   - Expected Dice: 0.75-0.82

2. **TransUNet** (Transformer-based)
   - Designed for medical imaging
   - Efficient for small datasets
   - Expected Dice: 0.72-0.78

3. **Task Simplification**
   - Binary task: "tumor vs non-tumor" (easier)
   - Two-stage: Segment first, classify second
   - May improve clinical utility

---

## Key Metrics to Track

**During Training**:
- ✅ Per-class Dice (tumor, stroma, lymphocyte, necrosis, blood_vessel)
- ✅ Training loss components (mask, dice, iou, class)
- ✅ Learning rate schedule
- ✅ GPU memory usage

**After Training**:
- ✅ Test per-class Dice/IoU
- ✅ Confusion matrix (which classes confused?)
- ✅ Per-image variance (consistent or erratic?)
- ✅ Small object detection (blood vessels)

**Success Criteria**:
- 🎯 Per-class Dice ≥ 0.65 (clinical utility threshold)
- 🎯 Blood vessel Dice ≥ 0.45 (hardest class)
- 🎯 CLIP accuracy ≥ 0.60 (useful for triage)
- 🎯 Stable training (no collapse)

---

## Commands Reference

```bash
# Test v3 config locally (recommended before HPRC)
bash scripts/training/test_v3_config.sh

# Run v3 training (HPRC)
sbatch scripts/slurm/run_training_v3.slurm

# Per-class validation (during or after training)
python scripts/validation/validate_perclass.py \
  --checkpoint finetune_logs/base_finetune_v3_*/checkpoints/checkpoint_15.pt \
  --model_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --split val \
  --output results/v3_validation.json

# Full evaluation (with CLIP)
python src/evaluation.py \
  --sam_checkpoint finetune_logs/base_finetune_v3_*/checkpoints/checkpoint_best.pt \
  --sam_model_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --clip_prompts configs/prompts/hard_coded_prompts_v2.json \
  --output_dir results/v3_evaluation

# Cleanup duplicates
bash scripts/cleanup.sh
```

---

## Notes

- v2 achieved 0.82 Dice in training but **0.42 real performance** - validation was wrong
- Zero-shot baseline running (will show if finetuning helped at all)
- Histopathology is hard: even experts struggle with boundaries
- Class imbalance is severe (blood vessels 2% vs tumor 27%)
- SAM needs significant adaptation for medical imaging
