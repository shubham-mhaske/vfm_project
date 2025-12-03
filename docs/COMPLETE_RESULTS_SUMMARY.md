# Complete Experiment Results Summary

## Overview

This document summarizes all experimental results from the VFM (Vision Foundation Models) project for medical image segmentation on the BCSS (Breast Cancer Semantic Segmentation) dataset.

**Dataset**: 151 H&E stained breast cancer images (45 test images, 5 classes)
**Evaluation Date**: December 2, 2024

---

## 1. SAM2 Segmentation Results (Zero-Shot)

### Complete Metrics by Prompt Strategy

| Strategy | Dice | IoU | Dice Std | IoU Std |
|----------|------|-----|----------|---------|
| **Box + Neg Points** | **0.555** | **0.408** | 0.193 | 0.185 |
| Box Baseline | 0.553 | 0.407 | 0.195 | 0.188 |
| Multi-Point | 0.418 | 0.287 | 0.209 | 0.178 |
| Centroid | 0.338 | 0.236 | 0.263 | 0.212 |

### Per-Class Performance (Box + Neg Points - Best Config)

| Class | Dice | IoU | Dice Std | IoU Std | Count |
|-------|------|-----|----------|---------|-------|
| Necrosis | **0.691** | **0.559** | 0.194 | 0.210 | 23 |
| Tumor | 0.560 | 0.403 | 0.147 | 0.139 | 45 |
| Stroma | 0.538 | 0.385 | 0.166 | 0.158 | 45 |
| Lymphocyte | 0.532 | 0.391 | 0.218 | 0.191 | 37 |
| Blood Vessel | 0.497 | 0.357 | 0.208 | 0.195 | 31 |

**Key Finding**: Box prompts significantly outperform point-based prompts (+21.7% Dice over centroid). Negative points provide marginal improvement (+0.2% Dice).

---

## 2. MedSAM Segmentation Results

| Strategy | Dice | IoU | Dice Std | IoU Std |
|----------|------|-----|----------|---------|
| MedSAM Box + TTA | **0.536** | **0.389** | 0.191 | 0.178 |
| MedSAM Box | 0.522 | 0.375 | 0.189 | 0.174 |

### Per-Class Performance (MedSAM Box + TTA)

| Class | Dice | IoU | Count |
|-------|------|-----|-------|
| Necrosis | **0.647** | **0.501** | 23 |
| Tumor | 0.575 | 0.421 | 45 |
| Lymphocyte | 0.549 | 0.405 | 37 |
| Stroma | 0.505 | 0.356 | 45 |
| Blood Vessel | 0.427 | 0.291 | 31 |

**Comparison**: SAM2 Box+Neg outperforms MedSAM Box+TTA by **1.9% Dice** (0.555 vs 0.536)

---

## 3. CLIP Classification Results

### Performance by Prompt Configuration

| Configuration | Accuracy | Precision | Recall | F1 |
|---------------|----------|-----------|--------|-----|
| **llm_text_v3_fewshot** | **44.4%** | 0.369 | 0.422 | 0.338 |
| hardcoded_v2 | 42.2% | 0.447 | 0.405 | 0.312 |
| llm_text_v2_clip_friendly | 35.6% | 0.455 | 0.351 | 0.270 |
| llm_multimodal_v3_fewshot | 29.4% | 0.292 | 0.303 | 0.220 |
| hardcoded_v1 | 23.3% | 0.261 | 0.211 | 0.138 |
| llm_multimodal_v2_clip_friendly | 15.0% | 0.247 | 0.219 | 0.100 |
| llm_text_v1_jargon | 12.2% | 0.272 | 0.082 | 0.097 |
| llm_multimodal_v1 | 8.3% | 0.250 | 0.056 | 0.091 |

### Best Configuration Per-Class Breakdown (llm_text_v3_fewshot)

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| Tumor | **0.974** | 0.822 | **0.892** | 45 |
| Stroma | 0.619 | 0.289 | 0.394 | 45 |
| Blood Vessel | 0.254 | **1.000** | 0.405 | 30 |
| Lymphocyte | 0.000 | 0.000 | 0.000 | 37 |
| Necrosis | 0.000 | 0.000 | 0.000 | 23 |

**Key Finding**: LLM-generated few-shot prompts perform best (44.4%), but classification struggles with lymphocyte/necrosis classes (0% recall).

---

## 4. Finetuned Model Results

### Training Runs Completed

| Model | Epochs | Prompt Type | Loss | Status |
|-------|--------|-------------|------|--------|
| SAM2 Box Focal | 50 | Box | Focal | ✅ Completed |
| SAM2 LoRA Light | 30 | Box | BCE+Dice | ✅ Completed |
| PathSAM2 CTransPath | 40 | Box | Focal | ✅ Completed |

### Evaluation Results

| Model | Overall Dice | Best Class | Worst Class |
|-------|--------------|------------|-------------|
| SAM2 Box Focal (Epoch 15) | **0.372** | Necrosis (0.478) | Lymphocyte (0.301) |
| SAM2 LoRA Light | 0.355 | Necrosis (0.498) | Lymphocyte (0.265) |
| PathSAM2 CTransPath Optimized | 0.016 | Tumor (0.024) | Blood Vessel (0.007) |

**Key Finding**: Finetuning with focal loss shows promise but underperforms zero-shot SAM2 (0.372 vs 0.555). PathSAM2 with CTransPath encoder failed to converge properly (0.016 Dice).

---

## 5. Comparative Analysis

### Segmentation Model Comparison

| Model | Dice Score | Configuration |
|-------|------------|---------------|
| **SAM2 (Zero-Shot)** | **0.555** | Box + Neg Points |
| MedSAM (Zero-Shot) | 0.536 | Box + TTA |
| SAM2 Box Focal (Finetuned) | 0.372 | Epoch 15 |
| SAM2 LoRA (Finetuned) | 0.355 | Epoch 30 |
| PathSAM2 CTransPath | 0.016 | Failed |

### Classification Model Performance

| Model | Accuracy | Best Strategy |
|-------|----------|---------------|
| CLIP (ViT-B/32) | **44.4%** | LLM Few-shot Prompts |
| CLIP (Hardcoded v2) | 42.2% | Manual Prompts |

---

## 6. Key Takeaways

### ✅ What Works
1. **SAM2 Zero-Shot** achieves best segmentation (0.555 Dice) with box prompts
2. **Box prompts** significantly outperform point-based prompts (+21.7%)
3. **LLM few-shot prompts** improve CLIP classification over hardcoded (+2.2%)
4. **MedSAM** performs competitively but slightly behind SAM2 (-1.9%)

### ⚠️ Challenges
1. **CLIP classification** struggles with minority classes (lymphocyte, necrosis)
2. **Finetuning** currently underperforms zero-shot (needs hyperparameter tuning)
3. **PathSAM2 with CTransPath** failed to converge (architecture mismatch issue)

### 📊 Per-Class Insights
- **Necrosis**: Easiest to segment (0.691 Dice) - distinctive texture/color
- **Tumor**: Best classified (97.4% precision) - most common and distinctive
- **Blood Vessel**: High recall (100%) but low precision (25.4%) - over-predicted
- **Lymphocyte/Necrosis**: Not classified at all - need better prompts or training data

---

## 7. Next Steps

1. **Hyperparameter tuning** for finetuning (learning rate, loss weights)
2. **Class-balanced sampling** to improve minority class performance
3. **Prompt engineering** specifically for lymphocyte/necrosis
4. **Ensemble methods** combining SAM2 + CLIP for end-to-end pipeline
5. **Fix PathSAM2** CTransPath encoder integration

---

## File Locations

| Category | Path |
|----------|------|
| Complete Metrics | `results/complete_metrics/` |
| SAM2 Metrics | `results/complete_metrics/sam2_segmentation_*.json` |
| CLIP Metrics | `results/complete_metrics/clip_classification_*.json` |
| MedSAM Metrics | `results/complete_metrics/medsam_segmentation_*.json` |
| Finetune Logs | `finetune_logs/*/logs/` |
| SLURM Logs | `slurm_logs/` |

---

*Generated: December 2, 2024*
