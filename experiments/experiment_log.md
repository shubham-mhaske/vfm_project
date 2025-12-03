# Experiment Log

**CSCE 689 - Visual Foundation Models for Medical Image Analysis (Fall 2025)**

Quick reference for experiment results. See `../EXPERIMENTS.md` for full details.

## 🎉 Key Findings

### Segmentation
**Zeroshot SAM2 with optimized prompts achieves the best results!**
- **Best Configuration**: Box + Negative Points + TTA = **0.550 Dice**

### Classification  
**CLIP with feature-based classifier outperforms zero-shot!**
- **Best Configuration**: CLIP + Logistic Regression = **40.4% Accuracy**

---

## Summary Tables

### Classification Results
| Method | Accuracy | Status |
|--------|----------|--------|
| **CLIP + LogReg** | **40.4%** | ✅ **BEST** |
| LLM Text Fewshot | 38.9% | ✅ Best zero-shot |
| CLIP Hardcoded v2 | 35.6% | ✅ |
| LLM Multimodal Fewshot | 30.3% | ⚠️ |
| PLIP Zero-shot | 26.9% | ⚠️ Worse than CLIP |
| CLIP Hardcoded v1 | 25.7% | Baseline |

### Segmentation Results
| Version | Date | Overall Dice | Status |
|---------|------|--------------|--------|
| **Box+Neg+TTA** | 2025-11-27 | **0.550** | ✅ **BEST** |
| Box+TTA | 2025-11-27 | 0.540 | ✅ |
| Box+NegPts | 2025-11-27 | 0.538 | ✅ |
| MedSAM Box+TTA | 2025-11-28 | 0.536 | ✅ |
| Box baseline | 2025-11-27 | 0.526 | ✅ |
| v2 Finetuned | 2025-11-22 | 0.42 | ❌ |
| Path-SAM2+CTP | 2025-11-27 | 0.366 | ❌ |
| LoRA-Light | 2025-11-27 | 0.355 | ❌ |
| LoRA r=8 | 2025-11-27 | 0.266 | ❌ Worst |

---

## Best Per-Class Results

### Classification (CLIP + LogReg)
| Class | Recall | F1 |
|-------|--------|-----|
| Stroma | 55.6% | 48.5% |
| Lymphocyte | 51.4% | 44.7% |
| Tumor | 28.9% | 32.5% |
| Necrosis | 26.1% | 32.4% |
| Blood Vessel | 0.0% | 0.0% |

### Segmentation (SAM2 + Box + TTA)
| Class | Dice |
|-------|------|
| Necrosis | 0.708 |
| Tumor | 0.565 |
| Lymphocyte | 0.562 |
| Stroma | 0.540 |
| Blood Vessel | 0.504 |

---

## Quick Commands

```bash
# Best Segmentation (0.550 Dice)
python src/evaluate_segmentation.py \
  --model_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --checkpoint sam2/checkpoints/sam2.1_hiera_large.pt \
  --prompt_type box --use_neg_points --use_tta \
  --split test --output_dir results/best_config

# Best Classification (40.4% Accuracy)
python src/train_clip_classifier.py \
  --output_dir results/clip_classifier

# Full Pipeline (SAM2 + CLIP)
python src/evaluation.py \
  --sam_model_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --sam_checkpoint sam2/checkpoints/sam2.1_hiera_large.pt \
  --clip_prompts configs/prompts/llm_text_prompts_v3_fewshot.json \
  --output_dir results/full_pipeline

# Generate Presentation Figures
python scripts/plot_training_curves.py
```

---

## Key Insights

### What Worked
1. **Box prompts >> Point prompts** (0.52 vs 0.33-0.42 Dice)
2. **Negative points add ~2%** (helps SAM know what NOT to segment)
3. **TTA adds ~2%** (free performance, no training)
4. **CLIP prompt engineering adds +13%** (fewshot prompts best)
5. **Feature classifier adds +1.5%** (LogReg on CLIP features)

### What Failed
1. **ALL finetuning hurt** - catastrophic forgetting with 85 images
2. **PLIP underperformed CLIP** - may need different prompts
3. **Ensemble methods hurt** - CLIP+PLIP worse than CLIP alone
4. **Multi-scale hurt** - more context confused classifier
5. **Blood vessel is hardest** - tiny structures (0.5% of pixels)

---

**Last Updated**: December 2, 2025
