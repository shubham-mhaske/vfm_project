# Experiment Tracker: Promptable Pathology

**CSCE 689 - Visual Foundation Models for Medical Image Analysis (Fall 2025)**

## 🎉 Key Findings

### Segmentation: Prompt Engineering Beats Fine-tuning
**Zeroshot SAM2 with optimized prompts (Box + Negative Points + TTA) achieves the best results.**

### Classification: CLIP Feature Classifier is Best
**CLIP + Logistic Regression achieves 40.4% on 5 tissue classes (zero-shot baseline: 25.7%).**

---

## Summary Tables

### Classification Results
| Method | Accuracy | Δ vs Baseline |
|--------|----------|---------------|
| **🥇 CLIP + LogReg** | **40.4%** | **+14.7%** ✅ |
| 🥈 LLM Text Fewshot | 38.9% | +13.2% |
| 🥉 CLIP Hardcoded v2 | 35.6% | +9.9% |
| LLM Multimodal Fewshot | 30.3% | +4.6% |
| LLM Multimodal v2 | 28.9% | +3.2% |
| PLIP Zero-shot | 26.9% | +1.2% |
| CLIP Hardcoded v1 (baseline) | 25.7% | - |
| LLM Text Jargon | 22.9% | -2.8% |
| LLM Multimodal v1 | 18.4% | -7.3% |

### Segmentation Results

| Approach | Overall Dice | vs Baseline |
|----------|--------------|-------------|
| **🥇 Box + Neg Points + TTA** | **0.550** | **+4.6%** ✅ |
| 🥈 Box + TTA | 0.540 | +2.7% |
| 🥉 Box + Neg Points | 0.538 | +2.3% |
| MedSAM Box + TTA | 0.536 | +1.9% |
| Box baseline | 0.526 | - |
| MedSAM Box | 0.522 | -0.8% |
| Multi-point prompt | 0.418 | -20.5% |
| v2 Finetuned (best finetune) | 0.42 | -20.2% |
| Path-SAM2 + CTransPath | 0.366 | -30.4% |
| SAM2 Box+Focal | 0.372 | -29.3% |
| SAM2 LoRA-Light | 0.355 | -32.5% |
| Centroid (single point) | 0.335 | -36.3% |
| LoRA Adapter r=8 | 0.266 | -49.4% ❌ |

**Key Insights:**
1. **Box prompts >> Point prompts** (0.52+ vs 0.33-0.42)
2. **Negative points add ~2% improvement** (helps SAM understand what NOT to segment)
3. **TTA adds ~2% improvement** (ensemble of hflip, vflip, rot90)
4. **ALL finetuning approaches HURT performance** - catastrophic forgetting with only 85 training images
5. **CLIP prompt engineering adds +13%** - Fewshot prompts dramatically improve classification

---

## 📊 CLIP Classification Experiments (December 2025)

### Prompt Comparison Results

| Experiment | Prompt Type | Accuracy | Notes |
|------------|-------------|----------|-------|
| exp1_hardcoded_v1 | Basic | 25.7% | Baseline |
| exp1_hardcoded_v2 | Improved | 35.6% | Better prompts |
| exp3_llm_text_v1 | Jargon | 22.9% | Medical jargon hurt |
| exp3_llm_text_v2 | CLIP-friendly | 33.9% | Better |
| **exp3_llm_text_v3** | **Fewshot** | **38.9%** | **Best zero-shot** |
| exp5_multimodal_v1 | Basic | 18.4% | Worst |
| exp5_multimodal_v2 | CLIP-friendly | 28.9% | OK |
| exp5_multimodal_v3 | Fewshot | 30.3% | Good |

### Alternative Models Tested

| Model | Method | Accuracy | Notes |
|-------|--------|----------|-------|
| **CLIP + LogReg** | Feature classifier | **40.4%** | Best overall |
| CLIP + MLP | Feature classifier | 32.1% | Overfits |
| PLIP | Zero-shot | 26.9% | Medical CLIP - worse |
| CLIP + PLIP | Ensemble | 31.4% | Ensemble hurt |
| CLIP Multi-scale | 3 scales | 27.8% | More context hurt |

### CLIP Classifier Per-Class Results

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Tumor | 37.1% | 28.9% | 32.5% | 45 |
| Stroma | 43.1% | 55.6% | 48.5% | 45 |
| Lymphocyte | 39.6% | 51.4% | 44.7% | 37 |
| Necrosis | 42.9% | 26.1% | 32.4% | 23 |
| Blood Vessel | 0.0% | 0.0% | 0.0% | 6 |
| **Overall** | - | **40.4%** | 38.8% | 156 |

```bash
python src/evaluate_segmentation.py \
  --model_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --checkpoint sam2/checkpoints/sam2.1_hiera_large.pt \
  --prompt_type box \
  --use_neg_points \
  --use_tta \
  --split test \
  --output_dir results/best_config
```

**Per-Class Results (Best Config: Box + Neg Points + TTA)**:

| Class | Dice | Count |
|-------|------|-------|
| Necrosis | 0.708 | 23 |
| Tumor | 0.565 | 45 |
| Lymphocyte | 0.562 | 37 |
| Stroma | 0.540 | 45 |
| Blood Vessel | 0.504 | 31 |
| **Overall** | **0.566** | 181 |

---

## Prompt Engineering Results (Nov 27, 2025)

### Complete Comparison

| Rank | Configuration | Tumor | Stroma | Lymph | Necrosis | Blood V. | **Overall** |
|------|---------------|-------|--------|-------|----------|----------|-------------|
| 🥇 | Box+Neg+TTA | 0.565 | 0.540 | 0.562 | 0.708 | 0.504 | **0.566** |
| 🥈 | Box+TTA | 0.560 | 0.523 | 0.562 | 0.716 | 0.515 | **0.563** |
| 🥉 | Box+NegPts | 0.560 | 0.537 | 0.549 | 0.699 | 0.504 | **0.560** |
| 4 | Box baseline | 0.548 | 0.509 | 0.554 | 0.704 | 0.512 | **0.553** |
| 5 | Multi-point | 0.494 | 0.380 | 0.364 | 0.473 | 0.370 | **0.418** |
| 6 | Centroid | 0.270 | 0.331 | 0.307 | 0.514 | 0.339 | **0.335** |

### Key Observations

1. **Box prompts are essential**: Single point (centroid) fails badly (0.335 vs 0.553)
2. **Negative points help stroma most**: +2.8% improvement (0.509 → 0.537)
3. **TTA helps necrosis most**: +1.2% improvement (0.704 → 0.716)
4. **Blood vessel is hardest**: Best score is only 0.515 (very small structures)

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

## Why Finetuning Failed

All finetuning approaches performed WORSE than zeroshot SAM2:

**Root Cause Analysis:**
1. **Dataset size**: Only 85 training images vs SAM's billions of training samples
2. **Catastrophic forgetting**: Even LoRA with 0.88% trainable params causes forgetting
3. **Domain mismatch**: Histopathology has very different visual patterns than natural images
4. **Training-Eval mismatch**: Training losses don't match full SAM2 predictor behavior

**Lesson Learned**: For small datasets (<1000 images), invest in prompt engineering rather than finetuning.

---

## Finetuning Experiments (All Failed)

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

### LoRA Adapter r=8 ❌ WORST RESULT
**Date**: 2025-11-27  
**Config**: src/train_with_lora.py
- LoRA rank: 8, alpha: 8.0
- Trainable params: 1,997,060 (0.88% of 224M)
- Target modules: image_encoder only
- LR: 1e-4, epochs: 20
- Loss: SimpleDiceBCELoss

**Training Progress**:
- Best Val Dice: 0.8949 (epoch 4-8)
- Final Val Dice: 0.8922 (overfitting after epoch 8)

**Test Set Evaluation (full SAM2 predictor)**:

| Class | Dice | Std |
|-------|------|-----|
| Tumor | 0.410 | 0.138 |
| Stroma | 0.452 | 0.202 |
| Lymphocyte | 0.177 | 0.143 |
| Necrosis | 0.106 | 0.102 |
| Blood Vessel | 0.008 | 0.007 |
| **Overall** | **0.266** | - |

**Analysis**: 
- ⚠️ **Worst result of all experiments** (49% worse than zeroshot)
- Training showed high Val Dice (0.89) but used simplified backbone-only forward pass
- Full SAM2 predictor evaluation reveals LoRA weights hurt mask decoder's ability to generate masks
- Even 0.88% trainable params causes catastrophic forgetting

**Key Insight**: The training loss (backbone features → mask) doesn't match the evaluation pipeline (full SAM2 predictor with prompts). LoRA adapters modified the image encoder features in ways incompatible with the frozen mask decoder.

---

## MedSAM Evaluation (Nov 28, 2025)

### Background
**MedSAM** (Nature Communications 2024) is SAM fine-tuned on 1.57M medical image-mask pairs across 10 modalities. We tested if this medical pretraining helps histopathology segmentation.

### Configuration
- **Model**: SAM ViT-B (86M params) - smaller than SAM2 Hiera-L (224M params)
- **Checkpoint**: `medsam_vit_b.pth` (358MB)
- **Prompts**: Box-only (MedSAM doesn't support negative points)
- **TTA**: Horizontal flip, vertical flip, 90° rotation

### Results

| Configuration | Tumor | Stroma | Lymph | Necrosis | Blood V. | **Overall** |
|---------------|-------|--------|-------|----------|----------|-------------|
| MedSAM Box | 0.573 | 0.486 | 0.542 | 0.615 | 0.407 | **0.522** |
| MedSAM Box+TTA | 0.575 | 0.505 | 0.549 | 0.647 | 0.427 | **0.536** |
| SAM2 Box+Neg+TTA | 0.565 | 0.540 | 0.562 | 0.708 | 0.504 | **0.566** |

### Analysis

❌ **MedSAM performs worse than SAM2 on histopathology:**

| Method | Overall Dice | vs SAM2 Best |
|--------|--------------|--------------|
| SAM2 Box+Neg+TTA | 0.566 | - (baseline) |
| MedSAM Box+TTA | 0.536 | **-5.3%** |
| MedSAM Box | 0.522 | **-7.8%** |

**Why MedSAM underperforms:**
1. **Smaller model**: MedSAM uses ViT-B (86M) vs SAM2's Hiera-L (224M)
2. **Training domain mismatch**: MedSAM trained mostly on radiology (CT/MRI), not histopathology
3. **No negative points**: MedSAM only supports box prompts, can't use negative point refinement
4. **TTA helps**: +2.7% improvement, but still can't match SAM2

**Conclusion**: SAM2 with Box+Neg+TTA remains the best approach. MedSAM's medical pretraining doesn't transfer well to histopathology.

---

## Commands

### Best Configuration (Recommended)
```bash
# Best Segmentation: SAM2 with Box + Negative Points + TTA (0.550 Dice)
python src/evaluate_segmentation.py \
  --model_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --checkpoint sam2/checkpoints/sam2.1_hiera_large.pt \
  --prompt_type box \
  --use_neg_points \
  --use_tta \
  --split test \
  --output_dir results/best_zeroshot

# Best Classification: CLIP + LogReg (40.4% Accuracy)
python src/train_clip_classifier.py \
  --output_dir results/clip_classifier

# Full Pipeline: SAM2 + CLIP
python src/evaluation.py \
  --sam_model_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --sam_checkpoint sam2/checkpoints/sam2.1_hiera_large.pt \
  --clip_prompts configs/prompts/llm_text_prompts_v3_fewshot.json \
  --output_dir results/full_pipeline
```

### Generate Presentation Figures
```bash
python scripts/plot_training_curves.py
# Outputs to: presentation_data/figures/
```

### Alternative Experiments
```bash
# PLIP evaluation (26.9% - worse than CLIP)
python src/test_plip.py --output_dir results/plip_test

# CLIP + PLIP Ensemble (31.4% - ensemble hurt)
python src/test_ensemble.py --output_dir results/ensemble_test

# Multi-scale CLIP (27.8% - more context hurt)
python src/test_multiscale.py --output_dir results/multiscale_test
```

### MedSAM Evaluation
```bash
# MedSAM Box + TTA (0.536 Dice - 2.6% worse than SAM2)
python src/evaluate_medsam.py \
  --checkpoint models/medsam_checkpoints/medsam_vit_b.pth \
  --split test --use_tta \
  --output_dir results/medsam_box_tta
```

---

## Key Lessons Learned

1. **Prompt Engineering > Finetuning** for small datasets (~85 images)
2. **Box prompts >> Point prompts** (0.52 vs 0.33 Dice)
3. **Negative points help** by telling SAM what NOT to segment (+2%)
4. **TTA is free performance** with no training required (+2%)
5. **CLIP prompt engineering** adds +13% classification accuracy
6. **Feature classifiers** add +1.5% over zero-shot CLIP
7. **Catastrophic forgetting** makes all finetuning counterproductive
8. **PLIP underperforms CLIP** on histopathology (may need different prompts)
9. **Blood vessel is hardest** due to tiny size (0.5% of pixels)

---

## Research References

1. **SAM2** (Meta AI): Segment Anything Model 2 with Hiera-L backbone
2. **CLIP** (OpenAI): Contrastive Language-Image Pre-training
3. **PLIP** (Pathology CLIP): CLIP fine-tuned on pathology images
4. **MedSAM** (Nature Communications): SAM fine-tuned on 1.57M medical images
5. **Medical SAM Adapter** (arXiv:2304.12620): Lightweight adapters for medical SAM
6. **SAM-Med2D** (arXiv:2308.16184): Fine-tuned on 4.6M medical images

**Key insight**: Successful medical SAM adaptations use 1.5M-4.6M images. With only 85 images, prompt engineering is the only viable path.

---

**Last Updated**: December 2, 2025
