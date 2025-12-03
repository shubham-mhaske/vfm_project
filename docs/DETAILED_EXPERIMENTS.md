# Detailed Experiment Documentation

**CSCE 689 - Visual Foundation Models for Medical Image Analysis (Fall 2025)**

This document provides comprehensive documentation of all experiments conducted in this project, including methodology, configurations, results, and analysis.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset Description](#2-dataset-description)
3. [Experimental Framework](#3-experimental-framework)
4. [Segmentation Experiments](#4-segmentation-experiments)
   - 4.1 [SAM2 Prompt Engineering](#41-sam2-prompt-engineering)
   - 4.2 [MedSAM Evaluation](#42-medsam-evaluation)
   - 4.3 [Fine-tuning Experiments](#43-fine-tuning-experiments)
5. [Classification Experiments](#5-classification-experiments)
   - 5.1 [CLIP Zero-Shot Classification](#51-clip-zero-shot-classification)
   - 5.2 [LLM-Generated Prompts](#52-llm-generated-prompts)
   - 5.3 [Feature-Based Classifiers](#53-feature-based-classifiers)
   - 5.4 [Alternative Models](#54-alternative-models)
6. [Key Findings & Lessons Learned](#6-key-findings--lessons-learned)
7. [Reproduction Guide](#7-reproduction-guide)

---

## 1. Project Overview

### 1.1 Research Question
Can Visual Foundation Models (VFMs) achieve competitive performance on histopathology segmentation and classification tasks without domain-specific fine-tuning?

### 1.2 Hypothesis
For small medical imaging datasets, prompt engineering will outperform fine-tuning due to:
- Limited training data (85 images) vs VFM pretraining scale (billions of images)
- Risk of catastrophic forgetting
- Domain gap between natural images and H&E stained tissue

### 1.3 Core Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    ZERO-SHOT PIPELINE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input Image ──► SAM2 ──► Segmentation Masks ──► Crop Regions  │
│       (H&E)     (Box+Neg+TTA)                                   │
│                                                                 │
│                              │                                  │
│                              ▼                                  │
│                                                                 │
│              Cropped Regions ──► CLIP ──► Class Labels          │
│                                 (Fewshot Prompts)               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.4 Models Used

| Model | Architecture | Parameters | Purpose |
|-------|--------------|------------|---------|
| **SAM2** | Hiera-L Transformer | 224M | Segmentation |
| **MedSAM** | ViT-B (SAM) | 86M | Medical segmentation baseline |
| **CLIP** | ViT-B/32 | 151M | Zero-shot classification |
| **PLIP** | ViT-B/32 (fine-tuned) | 151M | Pathology-specific classification |
| **CTransPath** | Swin Transformer | 28M | Pathology feature extraction |

---

## 2. Dataset Description

### 2.1 BCSS Dataset

**Breast Cancer Semantic Segmentation (BCSS)** is a crowd-sourced histopathology dataset for semantic segmentation of H&E-stained breast cancer tissue.

| Property | Value |
|----------|-------|
| **Source** | NuCLS (Amgad et al., 2019) |
| **Modality** | H&E stained histopathology |
| **Resolution** | 1024 × 1024 pixels |
| **Total Images** | 151 |
| **Train/Val/Test Split** | 85 / 21 / 45 |

### 2.2 Class Distribution

| Class | ID | Pixel Frequency | Weight Factor |
|-------|-----|-----------------|---------------|
| Background | 0 | 26.3% | - |
| **Tumor** | 1 | 33.8% | 1.0× (baseline) |
| **Stroma** | 2 | 26.2% | 1.3× |
| **Lymphocyte** | 3 | 5.9% | 5.7× |
| **Necrosis** | 4 | 6.9% | 4.9× |
| **Blood Vessel** | 18 | 0.5% | 67.6× |

**⚠️ Critical Note**: Blood vessel is class ID **18**, not 5. Class 5 (`glandular_secretions`) is excluded from evaluation.

### 2.3 Visual Characteristics

| Class | Visual Description |
|-------|-------------------|
| **Tumor** | Dense clusters of large, irregularly shaped purple cells with dark nuclei. High cell density, chaotic arrangement. |
| **Stroma** | Pink wavy fibrous tissue with scattered elongated nuclei. Collagen bundles running in parallel patterns. |
| **Lymphocyte** | Tiny, uniform dark blue-purple dots. Small round cells with minimal cytoplasm, often clustered. |
| **Necrosis** | Pale, ghostly pink tissue with faded, blurry appearance. Loss of cell structure, fragmented debris. |
| **Blood Vessel** | Circular hollow spaces with thin pink walls. Tube-like structures with smooth lining. |

### 2.4 Dataset Challenges

1. **Class Imbalance**: Blood vessel represents only 0.5% of pixels
2. **Visual Similarity**: Tumor and lymphocytes both appear as dark purple cells
3. **Small Dataset**: Only 85 training images limits fine-tuning potential
4. **Annotation Quality**: Crowd-sourced annotations may have inter-annotator variability

---

## 3. Experimental Framework

### 3.1 Evaluation Metrics

#### Segmentation Metrics
- **Dice Coefficient**: $\text{Dice} = \frac{2|A \cap B|}{|A| + |B|}$
- **IoU (Jaccard Index)**: $\text{IoU} = \frac{|A \cap B|}{|A \cup B|}$
- **Per-class and Overall** with standard deviation

#### Classification Metrics
- **Accuracy**: Overall correct predictions / total predictions
- **Precision**: $\frac{TP}{TP + FP}$ per class
- **Recall**: $\frac{TP}{TP + FN}$ per class
- **F1-Score**: Harmonic mean of precision and recall
- **Macro Average**: Unweighted mean across classes

### 3.2 Hardware & Environment

| Resource | Specification |
|----------|---------------|
| **GPU** | NVIDIA A100 (40GB) on TAMU HPRC Grace |
| **CPU** | 16-48 cores |
| **Memory** | 64-256 GB |
| **Framework** | PyTorch 2.0+ |
| **CLIP Model** | openai/clip-vit-base-patch32 |

### 3.3 Code Organization

```
src/
├── dataset.py              # BCSSDataset loader with deterministic splits
├── sam_segmentation.py     # SAM2 predictor helpers, prompts, metrics
├── clip_classification.py  # CLIP classifier, prompt loading
├── evaluation.py           # Full pipeline (SAM2 + CLIP)
├── evaluate_segmentation.py # SAM2-only evaluation
├── evaluate_medsam.py      # MedSAM evaluation
├── train_clip_classifier.py # Feature-based CLIP classifier
├── train_sam.py            # SAM2 fine-tuning (Hydra)
└── train_with_lora.py      # LoRA adapter training
```

---

## 4. Segmentation Experiments

### 4.1 SAM2 Prompt Engineering

#### 4.1.1 Experiment Design

We systematically evaluated SAM2's segmentation performance across different prompting strategies:

| Configuration | Prompt Type | Negative Points | Test-Time Augmentation |
|---------------|-------------|-----------------|------------------------|
| **Centroid** | Single point at mask center | ❌ | ❌ |
| **Multi-point** | 5 points sampled from mask | ❌ | ❌ |
| **Box Baseline** | Bounding box around mask | ❌ | ❌ |
| **Box + Neg** | Bounding box + background points | ✅ | ❌ |
| **Box + TTA** | Bounding box + augmentation | ❌ | ✅ |
| **Box + Neg + TTA** | All techniques combined | ✅ | ✅ |

#### 4.1.2 Prompt Generation Details

**Centroid Point**:
```python
centroid = np.array(np.where(mask > 0)).mean(axis=1)[::-1]  # [x, y]
```

**Multi-Point Sampling**:
```python
# Sample 5 points from inside the mask using distance transform
dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
# Select points with maximum distance from boundaries
```

**Bounding Box**:
```python
rows, cols = np.where(mask > 0)
box = [cols.min(), rows.min(), cols.max(), rows.max()]  # [x1, y1, x2, y2]
```

**Negative Points** (background sampling):
```python
# Sample 3 points from outside the mask but within box margin
background_mask = ~mask & dilated_box_region
neg_points = sample_points(background_mask, n=3)
```

**Test-Time Augmentation (TTA)**:
```python
# Apply augmentations and average predictions
augmentations = [identity, horizontal_flip, vertical_flip, rotate_90]
masks = [apply_aug_and_predict(aug) for aug in augmentations]
final_mask = (np.mean(masks, axis=0) > 0.5).astype(np.uint8)
```

#### 4.1.3 Results

**Per-Class Dice Scores** (Test Set, 45 images):

| Configuration | Tumor | Stroma | Lymphocyte | Necrosis | Blood V. | **Overall** |
|---------------|-------|--------|------------|----------|----------|-------------|
| Centroid | 0.270 | 0.331 | 0.307 | 0.514 | 0.339 | **0.335** |
| Multi-point | 0.494 | 0.380 | 0.364 | 0.473 | 0.370 | **0.418** |
| Box baseline | 0.548 | 0.509 | 0.554 | 0.704 | 0.512 | **0.526** |
| Box + Neg | 0.560 | 0.537 | 0.549 | 0.699 | 0.504 | **0.538** |
| Box + TTA | 0.560 | 0.523 | 0.562 | 0.716 | 0.515 | **0.540** |
| **Box + Neg + TTA** | 0.565 | 0.540 | 0.562 | 0.708 | 0.504 | **0.550** |

#### 4.1.4 Analysis

1. **Box prompts are essential**: Single point prompts fail badly (0.335 Dice) because they don't provide enough spatial context for SAM to understand the region of interest.

2. **Negative points help stroma most** (+2.8%): Stroma often borders tumor, and negative points help SAM distinguish the fibrous tissue from adjacent malignant cells.

3. **TTA helps necrosis most** (+1.2%): Necrotic regions have irregular, fragmented boundaries that benefit from multi-view ensembling.

4. **Blood vessel remains hardest**: Best score is only 0.504 Dice due to tiny structure size (often <50 pixels) and class imbalance.

5. **Combined approach wins**: Box + Neg + TTA achieves 0.550 Dice, a 4.6% improvement over baseline.

---

### 4.2 MedSAM Evaluation

#### 4.2.1 Background

**MedSAM** (Nature Communications 2024) is SAM fine-tuned on 1.57 million medical image-mask pairs across 10 imaging modalities:
- CT, MRI, X-ray, Ultrasound, Dermoscopy, Endoscopy, Fundus, Mammography, Pathology, etc.

We hypothesized MedSAM might outperform SAM2 on histopathology due to its medical pretraining.

#### 4.2.2 Configuration

| Parameter | Value |
|-----------|-------|
| **Model** | SAM ViT-B (86M params) |
| **Checkpoint** | `medsam_vit_b.pth` (358MB) |
| **Prompt Type** | Box only (MedSAM doesn't support negative points) |
| **TTA** | Horizontal flip, vertical flip, 90° rotation |

#### 4.2.3 Results

| Configuration | Tumor | Stroma | Lymphocyte | Necrosis | Blood V. | **Overall** |
|---------------|-------|--------|------------|----------|----------|-------------|
| MedSAM Box | 0.573 | 0.486 | 0.542 | 0.615 | 0.407 | **0.522** |
| MedSAM Box + TTA | 0.575 | 0.505 | 0.549 | 0.647 | 0.427 | **0.536** |
| SAM2 Box + Neg + TTA | 0.565 | 0.540 | 0.562 | 0.708 | 0.504 | **0.550** |

#### 4.2.4 Analysis

❌ **MedSAM underperforms SAM2 on histopathology**:

| Comparison | Dice | Difference |
|------------|------|------------|
| SAM2 Best vs MedSAM Best | 0.550 vs 0.536 | **-2.6%** |
| SAM2 Best vs MedSAM Baseline | 0.550 vs 0.522 | **-5.1%** |

**Why MedSAM underperforms**:
1. **Smaller architecture**: ViT-B (86M) vs Hiera-L (224M) — less capacity
2. **Training domain mismatch**: MedSAM trained mostly on radiology (CT/MRI), pathology was minor
3. **No negative points**: MedSAM only supports box prompts
4. **Older SAM architecture**: SAM2's Hiera backbone is more efficient than original ViT

**Conclusion**: SAM2 with prompt engineering remains superior. Medical pretraining doesn't automatically transfer to all medical imaging domains.

---

### 4.3 Fine-tuning Experiments

We conducted extensive fine-tuning experiments to test whether adapting SAM2 to histopathology would improve performance.

#### 4.3.1 Experiment v1: Base Fine-tuning (FAILED)

**Date**: November 21, 2025

| Parameter | Value |
|-----------|-------|
| Learning Rate | 2e-5 (no warmup) |
| Loss Weights | Mask: 20, Dice: 1, IoU: 1, Class: 1 |
| Epochs | 150 |
| Batch Size | 4 |
| Frozen | Encoder |

**Result**: Training collapsed at epoch 50 with Dice dropping to 0.07. Unstable gradients due to no warmup and aggressive learning rate.

---

#### 4.3.2 Experiment v2: Stable Fine-tuning (BEST FINETUNED)

**Date**: November 22, 2025

| Parameter | Value |
|-----------|-------|
| Learning Rate | 6e-5 → 6e-6 (cosine decay) |
| Warmup | 175 steps |
| Loss Weights | Mask: 5, Dice: 2, IoU: 1, Class: 1 |
| Epochs | 100 |
| Batch Size | 6 |
| Augmentation | ColorJitter(0.2), Rotation(90°) |
| Frozen | Encoder |

**Results (Epoch 15, Test Set)**:

| Class | Dice |
|-------|------|
| Tumor | 0.54 |
| Stroma | 0.43 |
| Lymphocyte | 0.20 |
| Necrosis | 0.45 |
| Blood Vessel | 0.03 |
| **Overall** | **0.42** |

**Analysis**: Stable training but still 20% worse than zero-shot (0.42 vs 0.526). Lymphocyte and blood vessel performance catastrophically degraded.

---

#### 4.3.3 Experiment v3: Per-Class Focused (WORSE)

**Date**: November 26, 2025

| Parameter | Value |
|-----------|-------|
| Learning Rate | 8e-5 |
| Warmup | 250 steps |
| Augmentation | ColorJitter(0.3) — stronger |
| Epochs | 40 |
| Class Weighting | Focal loss |

**Result**: Overall Dice 0.404 (-4% vs v2, -22% vs zero-shot). Stronger augmentation was too aggressive for H&E features.

---

#### 4.3.4 Path-SAM2 + CTransPath (FAILED)

**Date**: November 27, 2025

**Approach**: Combine SAM2 with CTransPath pathology encoder using attention fusion:

```
Image ──► CTransPath ──► Pathology Features ──┐
                                              │ Attention Fusion ──► SAM2 Decoder
Image ──► SAM2 Encoder ──► General Features ──┘
```

**Results**:

| Class | Dice | Std |
|-------|------|-----|
| Tumor | 0.345 | 0.215 |
| Stroma | 0.388 | 0.246 |
| Lymphocyte | 0.330 | 0.262 |
| Necrosis | 0.530 | 0.309 |
| Blood Vessel | 0.283 | 0.241 |
| **Overall** | **0.366** | - |

**Analysis**: 30% worse than zero-shot. CTransPath features may be incompatible with SAM2's architecture, or fusion mechanism wasn't optimal.

---

#### 4.3.5 LoRA Adapter Training (WORST)

**Date**: November 27, 2025

**Approach**: Low-Rank Adaptation to minimize trainable parameters and reduce forgetting:

| Parameter | Value |
|-----------|-------|
| LoRA Rank | 8 |
| LoRA Alpha | 8.0 |
| Trainable Params | 1,997,060 (0.88% of 224M) |
| Target Modules | Image encoder only |
| Learning Rate | 1e-4 |
| Epochs | 20 |
| Loss | SimpleDiceBCELoss |

**Training Metrics**:
- Best Validation Dice: 0.8949 (epoch 4-8)
- Final Validation Dice: 0.8922

**Test Set Evaluation** (Full SAM2 predictor):

| Class | Dice | Std |
|-------|------|-----|
| Tumor | 0.410 | 0.138 |
| Stroma | 0.452 | 0.202 |
| Lymphocyte | 0.177 | 0.143 |
| Necrosis | 0.106 | 0.102 |
| Blood Vessel | 0.008 | 0.007 |
| **Overall** | **0.266** | - |

**Analysis**: 
- ⚠️ **Worst result** — 49% worse than zero-shot
- Training showed high validation Dice (0.89) but used simplified forward pass
- Full SAM2 predictor evaluation reveals LoRA weights corrupted mask decoder compatibility
- Even 0.88% trainable parameters caused catastrophic forgetting

**Critical Insight**: Training loss (backbone features → direct mask) doesn't match evaluation pipeline (full SAM2 predictor with prompts). The LoRA-adapted encoder produces features that the frozen mask decoder cannot interpret correctly.

---

#### 4.3.6 Fine-tuning Summary

| Experiment | Overall Dice | vs Zero-shot |
|------------|--------------|--------------|
| Zero-shot SAM2 (Box) | 0.526 | Baseline |
| v2 Stable Finetune | 0.420 | **-20%** |
| v3 Per-class | 0.404 | **-23%** |
| SAM2 LoRA Light | 0.355 | **-33%** |
| Path-SAM2 + CTransPath | 0.366 | **-30%** |
| LoRA r=8 | 0.266 | **-49%** |

**Conclusion**: ALL fine-tuning approaches hurt performance. With only 85 training images, the models forget their pretrained knowledge faster than they can learn domain-specific patterns.

---

## 5. Classification Experiments

### 5.1 CLIP Zero-Shot Classification

#### 5.1.1 Methodology

CLIP classifies cropped tissue regions by comparing image embeddings with text embeddings of class descriptions:

```python
# For each class, compute average similarity across all prompts
similarities = []
for class_name, prompts in prompt_dict.items():
    text_embeds = clip.encode_text(prompts)
    image_embed = clip.encode_image(cropped_region)
    similarity = cosine_similarity(image_embed, text_embeds.mean())
    similarities.append(similarity)
    
predicted_class = class_names[argmax(similarities)]
```

#### 5.1.2 Baseline Prompts (v1)

```json
{
    "tumor": [
        "a histopathology image of a tumor",
        "cancerous tissue with malignant cells",
        "a dense cluster of large, irregularly shaped cells with dark nuclei"
    ],
    "stroma": [
        "a histopathology image of stroma",
        "connective tissue supporting the tumor",
        "spindle-shaped cells with elongated nuclei"
    ]
    // ... etc
}
```

**Result**: 25.7% accuracy — CLIP doesn't understand medical jargon.

#### 5.1.3 Improved Prompts (v2)

Rewrote prompts to describe visual appearance in CLIP-friendly language:

```json
{
    "tumor": [
        "densely crowded dark purple cells packed together",
        "large irregular purple nuclei in chaotic arrangement",
        "thick masses of deep purple tissue with high cell density"
    ],
    "stroma": [
        "bright pink wavy fibers forming streaming patterns",
        "light pink collagen with scattered thin elongated nuclei",
        "parallel bundles of pink fibrous tissue"
    ]
    // ... etc
}
```

**Result**: 35.6% accuracy — +9.9% improvement by describing colors and patterns.

---

### 5.2 LLM-Generated Prompts

#### 5.2.1 Text-Only LLM Prompts

**Approach**: Use Gemini Pro to generate class descriptions optimized for CLIP.

**Version 1 (Jargon)**: Asked LLM to describe histopathology classes
- Result: 22.9% accuracy — worse than baseline! LLM used medical terminology.

**Version 2 (CLIP-friendly)**: Asked LLM to describe visual appearance for image matching
- Result: 33.9% accuracy — improved but not better than manual v2.

**Version 3 (Few-shot)**: Provided examples of good prompts and asked for more
- Result: **38.9% accuracy** — Best zero-shot result!

```json
// Example few-shot generated prompts
{
    "tumor": [
        "densely crowded dark purple cells packed together",
        "overlapping dark purple nuclei forming a solid sheet",
        "jumbled piles of large, intensely purple nuclei",
        "a dense aggregate of large, irregularly shaped purple cells"
    ]
}
```

#### 5.2.2 Multimodal LLM Prompts

**Approach**: Provide Gemini 2.5 Flash with example images from each class and ask for descriptions.

| Version | Approach | Accuracy |
|---------|----------|----------|
| v1 | Basic image description | 18.4% — Worst! |
| v2 | CLIP-friendly visual descriptions | 28.9% |
| v3 | Few-shot with good examples | 30.3% |

**Analysis**: Multimodal prompts underperformed text-only. The LLM described medical features rather than visual patterns that CLIP understands.

---

### 5.3 Feature-Based Classifiers

#### 5.3.1 Approach

Instead of zero-shot classification, extract CLIP features and train a simple classifier:

```python
# Extract features for all training samples
features = []
labels = []
for image, mask, class_id in train_dataset:
    crop = crop_region_from_mask(image, mask)
    feat = clip.encode_image(crop)
    features.append(feat)
    labels.append(class_id)

# Train classifier on CLIP features
classifier = LogisticRegression()
classifier.fit(features, labels)
```

#### 5.3.2 Results

| Classifier | Accuracy | Notes |
|------------|----------|-------|
| **Logistic Regression** | **40.4%** | Best overall |
| MLP (2 layers) | 32.1% | Overfits |
| SVM (RBF kernel) | 38.2% | Good |
| Random Forest | 35.8% | OK |

#### 5.3.3 Per-Class Results (LogReg)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Tumor | 37.1% | 28.9% | 32.5% | 45 |
| Stroma | 43.1% | 55.6% | 48.5% | 45 |
| Lymphocyte | 39.6% | 51.4% | 44.7% | 37 |
| Necrosis | 42.9% | 26.1% | 32.4% | 23 |
| Blood Vessel | 0.0% | 0.0% | 0.0% | 6 |
| **Macro Avg** | 32.5% | 32.4% | 31.6% | - |

**Analysis**: 
- Stroma and lymphocyte have best recall (visual features most distinctive)
- Blood vessel has 0% — too few samples (only 6 in test set)
- +1.5% over best zero-shot (40.4% vs 38.9%)

---

### 5.4 Alternative Models

#### 5.4.1 PLIP (Pathology Language-Image Pretraining)

**Hypothesis**: PLIP, trained on pathology images, should outperform general CLIP.

**Result**: 26.9% accuracy — worse than CLIP!

**Analysis**: PLIP may have been trained with different prompt styles, or its pathology knowledge doesn't transfer to our specific tissue types.

#### 5.4.2 CLIP + PLIP Ensemble

**Approach**: Average predictions from both models.

**Result**: 31.4% accuracy — ensemble hurt performance.

**Analysis**: PLIP's incorrect predictions diluted CLIP's correct ones.

#### 5.4.3 Multi-Scale CLIP

**Approach**: Classify at 3 scales (0.5x, 1x, 2x crop size) and average.

**Result**: 27.8% accuracy — more context hurt.

**Analysis**: Larger crops include adjacent tissue types, confusing the classifier.

---

### 5.5 Classification Summary

| Method | Accuracy | Δ vs Baseline |
|--------|----------|---------------|
| **CLIP + LogReg** | **40.4%** | **+14.7%** |
| LLM Text Fewshot | 38.9% | +13.2% |
| CLIP Hardcoded v2 | 35.6% | +9.9% |
| LLM Multimodal v3 | 30.3% | +4.6% |
| LLM Multimodal v2 | 28.9% | +3.2% |
| Multi-scale CLIP | 27.8% | +2.1% |
| PLIP Zero-shot | 26.9% | +1.2% |
| CLIP Hardcoded v1 | 25.7% | Baseline |
| LLM Text Jargon | 22.9% | -2.8% |
| LLM Multimodal v1 | 18.4% | -7.3% |

---

## 6. Key Findings & Lessons Learned

### 6.1 Main Results

1. **Prompt Engineering > Fine-tuning** for small datasets (~85 images)
   - Best zero-shot: 0.550 Dice, 40.4% Accuracy
   - Best fine-tuned: 0.420 Dice — 25% worse

2. **Box prompts are essential** for SAM2
   - Box: 0.526 Dice vs Centroid: 0.335 Dice (+57%)

3. **Negative points help** by telling SAM what NOT to segment
   - +2.3% improvement (0.526 → 0.538)

4. **TTA is free performance** with no training required
   - +2.7% improvement (0.526 → 0.540)

5. **Visual descriptions > Medical jargon** for CLIP
   - "dark purple cells" works; "malignant neoplasm" doesn't

6. **Feature classifiers outperform zero-shot** by small margin
   - +1.5% (40.4% vs 38.9%)

### 6.2 Why Fine-tuning Failed

| Factor | Impact |
|--------|--------|
| **Small dataset** | 85 images vs SAM's billions — severe underfitting |
| **Catastrophic forgetting** | Models forget pretrained knowledge |
| **Domain gap** | H&E staining very different from natural images |
| **Training/Eval mismatch** | Training loss ≠ full predictor behavior |

### 6.3 Recommendations

1. **For segmentation**: Use SAM2 with Box + Negative Points + TTA
2. **For classification**: Use CLIP + feature classifier (LogReg)
3. **For prompts**: Describe visual appearance, not medical terminology
4. **For small datasets**: Invest in prompt engineering, not fine-tuning
5. **For rare classes**: Consider oversampling or specialized prompts

---

## 7. Reproduction Guide

### 7.1 Environment Setup

```bash
# Clone repository
git clone <repo-url> && cd vfm_project
git submodule update --init --recursive

# Install dependencies
pip install -r requirements.txt

# Download checkpoints
bash sam2/checkpoints/download_ckpts.sh
# Download MedSAM checkpoint separately
```

### 7.2 Run Best Configurations

```bash
# Best Segmentation (0.550 Dice)
python src/evaluate_segmentation.py \
  --model_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --checkpoint sam2/checkpoints/sam2.1_hiera_large.pt \
  --prompt_type box \
  --use_neg_points \
  --use_tta \
  --split test \
  --output_dir results/best_segmentation

# Best Classification (40.4% Accuracy)
python src/train_clip_classifier.py \
  --output_dir results/best_classification

# Full Pipeline
python src/evaluation.py \
  --sam_model_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --sam_checkpoint sam2/checkpoints/sam2.1_hiera_large.pt \
  --clip_prompts configs/prompts/llm_text_prompts_v3_fewshot.json \
  --output_dir results/full_pipeline
```

### 7.3 HPRC Submission

```bash
# Submit all metrics collection jobs
bash scripts/slurm/submit_all_metrics.sh

# Monitor
squeue -u $USER
tail -f slurm_logs/*.out
```

---

## Appendix A: File Locations

| Resource | Path |
|----------|------|
| SAM2 Checkpoint | `sam2/checkpoints/sam2.1_hiera_large.pt` |
| MedSAM Checkpoint | `models/medsam_checkpoints/medsam_vit_b.pth` |
| CLIP Model | `$SCRATCH/clip_model` (HPRC) or HuggingFace |
| Dataset Images | `data/bcss/images/` |
| Dataset Masks | `data/bcss/masks/` |
| Prompt Configs | `configs/prompts/*.json` |
| Results | `results/*/metrics.json` |

## Appendix B: Class ID Mapping

```python
CLASS_NAMES = {
    0: 'background',
    1: 'tumor',
    2: 'stroma',
    3: 'lymphocyte',
    4: 'necrosis',
    18: 'blood_vessel'  # Note: NOT 5!
}

TARGET_CLASS_IDS = [1, 2, 3, 4, 18]
```

---

**Document Version**: 1.0  
**Last Updated**: December 2, 2025  
**Author**: VFM Project Team
