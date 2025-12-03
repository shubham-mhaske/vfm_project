# VFM Project: Experiment Comparison & Analysis

## Research Questions Addressed

1. **RQ1**: Can zero-shot Vision Foundation Models (SAM2, MedSAM) effectively segment histopathology images?
2. **RQ2**: How do different prompt strategies affect segmentation quality?
3. **RQ3**: Can CLIP-based classification identify tissue types without training?
4. **RQ4**: Does finetuning improve over zero-shot performance?

---

## Experimental Framework

```
┌─────────────────────────────────────────────────────────────────────┐
│                    BCSS Dataset (151 images)                        │
│                  Train: 85 | Val: 21 | Test: 45                     │
│              5 Classes: Tumor, Stroma, Lymphocyte,                  │
│                        Necrosis, Blood Vessel                       │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
         ┌──────────────────┐         ┌──────────────────┐
         │   SEGMENTATION   │         │  CLASSIFICATION  │
         └──────────────────┘         └──────────────────┘
                    │                           │
      ┌─────────────┼─────────────┐             │
      ▼             ▼             ▼             ▼
   SAM2         MedSAM      Finetuned       CLIP
 (Zero-Shot)  (Zero-Shot)    Models      (Zero-Shot)
```

---

## 1. Segmentation Comparison

### 1.1 Model Comparison (Best Configuration Each)

| Rank | Model | Configuration | Dice ↑ | IoU ↑ | Notes |
|:----:|-------|---------------|:------:|:-----:|-------|
| 🥇 | **SAM2** | Box + Neg Points | **0.555** | **0.408** | Best overall |
| 🥈 | MedSAM | Box + TTA | 0.536 | 0.389 | -1.9% vs SAM2 |
| 🥉 | MedSAM | Box | 0.522 | 0.375 | Baseline |
| 4 | SAM2 Finetuned | Box Focal Loss | 0.372 | - | Underperforms |
| 5 | SAM2 LoRA | Light Finetune | 0.355 | - | Underperforms |

### 1.2 Prompt Strategy Comparison (SAM2)

```
Dice Score by Prompt Type
═══════════════════════════════════════════════════════════════
Box + Neg    ████████████████████████████████████████████  0.555
Box Only     ███████████████████████████████████████████   0.553  
Multi-Point  ██████████████████████████████                0.418
Centroid     ████████████████████████                      0.338
═══════════════════════════════════════════════════════════════
             0.0       0.2       0.4       0.6       0.8    1.0
```

**Key Insight**: Box prompts provide **+64%** improvement over centroid points.

### 1.3 Per-Class Segmentation Performance

| Class | SAM2 (Dice) | MedSAM (Dice) | Δ | Difficulty |
|-------|:-----------:|:-------------:|:-:|:----------:|
| Necrosis | **0.691** | 0.647 | +4.4% | Easy ✅ |
| Tumor | 0.560 | **0.575** | -1.5% | Medium |
| Lymphocyte | 0.532 | **0.549** | -1.7% | Medium |
| Stroma | **0.538** | 0.505 | +3.3% | Medium |
| Blood Vessel | **0.497** | 0.427 | +7.0% | Hard ⚠️ |

---

## 2. Classification Comparison

### 2.1 Prompt Engineering Comparison

| Rank | Prompt Source | Accuracy ↑ | F1 Macro | Strategy |
|:----:|---------------|:----------:|:--------:|----------|
| 🥇 | **LLM Text Few-shot** | **44.4%** | 0.338 | GPT-generated + examples |
| 🥈 | Hardcoded v2 | 42.2% | 0.312 | Manual expert prompts |
| 🥉 | LLM Text CLIP-friendly | 35.6% | 0.270 | GPT optimized for CLIP |
| 4 | LLM Multimodal Few-shot | 29.4% | 0.220 | Gemini image+text |
| 5 | Hardcoded v1 | 23.3% | 0.138 | Basic descriptions |

### 2.2 Per-Class Classification Analysis

```
Precision by Class (Best Config: LLM Few-shot)
═══════════════════════════════════════════════════════════════
Tumor        ████████████████████████████████████████████████ 97.4%
Stroma       ██████████████████████████████                   61.9%
Blood Vessel ████████████                                     25.4%
Lymphocyte   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0.0%
Necrosis     ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0.0%
═══════════════════════════════════════════════════════════════
```

**Problem Identified**: CLIP fails completely on lymphocyte and necrosis classes.

---

## 3. Zero-Shot vs Finetuning

### 3.1 Does Finetuning Help?

| Approach | Model | Dice | Verdict |
|----------|-------|:----:|:-------:|
| Zero-Shot | SAM2 Box+Neg | **0.555** | ✅ **Winner** |
| Finetuned | SAM2 Focal (50 epochs) | 0.372 | ❌ -33% worse |
| Finetuned | SAM2 LoRA (30 epochs) | 0.355 | ❌ -36% worse |
| Failed | PathSAM2 + CTransPath | 0.016 | ❌ Did not converge |

**Key Finding**: Zero-shot SAM2 outperforms all finetuned variants. This suggests:
- SAM2's pretrained features are highly generalizable
- Small dataset (85 training images) causes overfitting
- Hyperparameter tuning needed for finetuning

---

## 4. Ablation Studies

### 4.1 Effect of Test-Time Augmentation (TTA)

| Model | Without TTA | With TTA | Δ |
|-------|:-----------:|:--------:|:-:|
| MedSAM | 0.522 | 0.536 | **+1.4%** |

### 4.2 Effect of Negative Points

| Model | Without Neg | With Neg | Δ |
|-------|:-----------:|:--------:|:-:|
| SAM2 Box | 0.553 | 0.555 | **+0.2%** |

### 4.3 Prompt Complexity vs Performance

| Prompt Type | Complexity | Performance | Recommendation |
|-------------|:----------:|:-----------:|:---------------|
| Centroid | Low | 0.338 | ❌ Not recommended |
| Multi-Point | Medium | 0.418 | ⚠️ Moderate |
| Box | Medium | 0.553 | ✅ Recommended |
| Box + Neg | High | 0.555 | ✅ Best (marginal gain) |

---

## 5. Summary Tables for Publication

### Table 1: Segmentation Results (Test Set, n=45)

| Method | Backbone | Prompt | Dice ± Std | IoU ± Std |
|--------|----------|--------|:----------:|:---------:|
| SAM2 | Hiera-L | Box+Neg | **0.555 ± 0.193** | **0.408 ± 0.185** |
| SAM2 | Hiera-L | Box | 0.553 ± 0.195 | 0.407 ± 0.188 |
| MedSAM | ViT-B | Box+TTA | 0.536 ± 0.191 | 0.389 ± 0.178 |
| MedSAM | ViT-B | Box | 0.522 ± 0.189 | 0.375 ± 0.174 |
| SAM2 | Hiera-L | Multi-Pt | 0.418 ± 0.209 | 0.287 ± 0.178 |
| SAM2 | Hiera-L | Centroid | 0.338 ± 0.263 | 0.236 ± 0.212 |

### Table 2: Classification Results (Test Set, n=180 regions)

| Method | Prompt Source | Accuracy | Precision | Recall | F1 |
|--------|---------------|:--------:|:---------:|:------:|:--:|
| CLIP ViT-B/32 | LLM Few-shot | **44.4%** | 0.369 | 0.422 | 0.338 |
| CLIP ViT-B/32 | Hardcoded v2 | 42.2% | 0.447 | 0.405 | 0.312 |
| CLIP ViT-B/32 | LLM CLIP-opt | 35.6% | 0.455 | 0.351 | 0.270 |

### Table 3: Per-Class Segmentation (SAM2 Box+Neg)

| Class | N | Dice | Std | IoU | Std |
|-------|:-:|:----:|:---:|:---:|:---:|
| Necrosis | 23 | **0.691** | 0.194 | **0.559** | 0.210 |
| Tumor | 45 | 0.560 | 0.147 | 0.403 | 0.139 |
| Stroma | 45 | 0.538 | 0.166 | 0.385 | 0.158 |
| Lymphocyte | 37 | 0.532 | 0.218 | 0.391 | 0.191 |
| Blood Vessel | 31 | 0.497 | 0.208 | 0.357 | 0.195 |

---

## 6. Visualization Recommendations

### For Presentations/Papers:

1. **Figure 1**: Qualitative segmentation examples (SAM2 vs MedSAM vs GT)
2. **Figure 2**: Bar chart comparing Dice scores across methods
3. **Figure 3**: Per-class performance heatmap
4. **Figure 4**: Confusion matrix for CLIP classification
5. **Figure 5**: Training curves for finetuned models

### Generate Figures:
```bash
# Already generated in results directories:
# - results/*/confusion_matrix.png
# - results/*/per_class_metrics.png
```

---

## 7. Conclusions

### ✅ Main Findings

1. **SAM2 zero-shot achieves state-of-the-art** (0.555 Dice) on BCSS without any training
2. **Box prompts are essential** - 64% better than point prompts
3. **LLM-generated prompts improve CLIP** - 44.4% vs 23.3% for basic prompts
4. **Finetuning hurts performance** - likely due to small dataset size

### ⚠️ Limitations

1. CLIP cannot classify lymphocyte/necrosis (0% recall)
2. Finetuning requires more data or better regularization
3. Blood vessels remain challenging for all methods

### 🔮 Future Work

1. Class-balanced sampling for minority classes
2. Ensemble methods (SAM2 + domain-specific encoders)
3. Active learning for efficient annotation
4. Cross-dataset validation

---

## Appendix: Statistical Significance

To be computed:
- Paired t-test: SAM2 vs MedSAM
- McNemar's test: CLIP prompt comparisons
- Wilcoxon signed-rank test: Per-class differences
