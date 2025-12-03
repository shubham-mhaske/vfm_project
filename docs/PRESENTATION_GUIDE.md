# 6-Minute Academic Presentation Guide

## 📊 PUBLICATION-QUALITY FIGURES

Location: `results/figures/academic/`

| Figure | Description | Recommended Slide |
|--------|-------------|-------------------|
| **fig4_method_overview.png** | Pipeline architecture schematic | Slide 3: Methods |
| **fig1_segmentation_comprehensive.png** | 4-panel segmentation analysis | Slide 5-6: Main Results ⭐ |
| **fig2_clip_analysis.png** | 3-panel CLIP classification | Slide 7: CLIP Results ⭐ |
| **fig3_training_analysis.png** | 3-panel training curves | Slide 8: Why Finetuning Failed ⭐ |
| **fig5_summary_results.png** | Complete results tables | Slide 9: Summary |

---

## ⏱️ 12-SLIDE STRUCTURE (6 minutes)

| Slide | Time | Figure | Key Message |
|-------|------|--------|-------------|
| 1. Title | 10s | Sample image | VFMs for Medical Image Segmentation |
| 2. Problem | 30s | - | Manual annotation costly; VFMs pretrained on billions |
| 3. Methods | 40s | `fig4_method_overview.png` | Two-stage: SAM2→CLIP pipeline |
| 4. Dataset | 20s | - | BCSS: 151 images, 5 classes, 45 test |
| **5-6. Segmentation** | **60s** | **`fig1_segmentation_comprehensive.png`** | **Box+Neg best (0.555), finetuning hurts (-33%)** |
| **7. CLIP** | **40s** | **`fig2_clip_analysis.png`** | **Few-shot LLM best (44.4%), text > multimodal** |
| **8. Training** | **40s** | **`fig3_training_analysis.png`** | **Overfitting evidence, catastrophic forgetting** |
| 9. Summary | 30s | `fig5_summary_results.png` | All results in tables |
| 10. Key Findings | 30s | - | 4 takeaways |
| 11. Conclusion | 20s | - | Best config: SAM2 Box+Neg + LLM Few-Shot |
| 12. Thank You | 10s | - | Questions |

---

## 🔑 KEY NUMBERS TO MEMORIZE

| Metric | Value | Context |
|--------|-------|---------|
| **0.555** | Dice | SAM2 zero-shot (Box+Neg) - BEST |
| **-33%** | Drop | Finetuning vs zero-shot |
| **+64%** | Gain | Box vs point prompts |
| **44.4%** | Accuracy | CLIP with LLM few-shot prompts |
| **+264%** | Gain | Prompt evolution v1→v3 |

---

## 📋 WHAT EACH FIGURE SHOWS

### Fig 1: Segmentation Comprehensive (4 panels)
```
┌─────────────────────────────────────────────────────────────────────┐
│ (a) Prompt Ablation        │ (b) Model Comparison                   │
│ • Centroid: 0.338          │ • SAM2 Box+Neg: 0.555 (best)          │
│ • Multi-Pt: 0.418          │ • SAM2 Box: 0.553                     │
│ • Box: 0.553               │ • MedSAM+TTA: 0.536                   │
│ • Box+Neg: 0.555 (+64%)    │ • MedSAM Box: 0.522                   │
│                            │ • p < 0.05 significance               │
├────────────────────────────┼────────────────────────────────────────┤
│ (c) Zero-Shot vs Finetuned │ (d) Per-Class Heatmap                 │
│ • Zero-shot: 0.555         │ • Necrosis: 0.69 (easiest)            │
│ • Focal 50ep: 0.372 (-33%) │ • Tumor: 0.56                         │
│ • BCE 100ep: 0.371 (-33%)  │ • Stroma: 0.54                        │
│ • LoRA 30ep: 0.355 (-36%)  │ • Blood Vessel: 0.50 (hardest)        │
└─────────────────────────────────────────────────────────────────────┘
```

### Fig 2: CLIP Analysis (3 panels)
```
┌─────────────────────────────────────────────────────────────────────┐
│ (a) Strategy Comparison    │ (b) Per-Class    │ (c) Evolution       │
│ • LLM Few-Shot: 44.4%     │ Per-class acc    │ v1 Jargon: 12.2%   │
│ • Manual v2: 42.2%        │ for 3 methods    │ v2 Optimized: 35.6%│
│ • LLM Text v2: 35.6%      │ across 5 classes │ v3 Few-Shot: 44.4% │
│ • LLM VLM: 29.4%          │                  │ (+264% gain)        │
│ • Manual v1: 23.3%        │                  │                     │
│ • LLM VLM v1: 8.3%        │                  │ Text > Multimodal   │
└─────────────────────────────────────────────────────────────────────┘
```

### Fig 3: Training Analysis (3 panels)
```
┌─────────────────────────────────────────────────────────────────────┐
│ (a) Training Loss          │ (b) Validation Dice  │ (c) Final Test │
│ • Rapid convergence       │ • Peak at epoch 8    │ • Zero-shot best│
│ • All methods → 0.1       │ • Then declines      │ • All finetuned │
│ • Classic overfitting     │ • Overfitting region │   worse by 33%+ │
│   pattern                 │   shaded             │                 │
└─────────────────────────────────────────────────────────────────────┘
```

### Fig 4: Method Overview
- Complete pipeline schematic
- Stage 1: Input → SAM2/MedSAM (with prompts) → Binary Mask
- Stage 2: Crop → CLIP (with text prompts) → Class Label
- Model parameters and dataset info annotated

### Fig 5: Summary Tables
- Table (a): All segmentation configurations with Dice/IoU
- Table (b): Finetuning methods comparison
- Table (c): All CLIP prompt strategies

---

## 🗣️ 60-SECOND SPEAKING SCRIPTS

### Slide 5-6 (Segmentation - 60s):
"This figure shows our complete segmentation analysis. Panel A demonstrates that box prompts dramatically outperform point prompts - a 64% improvement in Dice score. Panel B compares SAM2 and MedSAM, with SAM2 achieving the best performance at 0.555 Dice. Critically, Panel C shows that all finetuning attempts hurt performance by 33% compared to zero-shot. Panel D breaks down per-class performance, showing necrosis is easiest and blood vessels most challenging."

### Slide 7 (CLIP - 40s):
"For CLIP classification, we tested 8 prompt strategies. The key finding is that LLM-generated prompts with few-shot examples achieved 44.4% accuracy - the best result. Panel C shows the evolution from medical jargon at 12% to optimized visual language at 44% - a 264% improvement. Importantly, text-only LLM beat multimodal, because images triggered medical vocabulary CLIP doesn't understand."

### Slide 8 (Training - 40s):
"Why did finetuning fail? Panel A shows training loss converging rapidly - a sign of overfitting to our small 85-image dataset. Panel B reveals validation Dice peaks at epoch 8 then declines as the model forgets pretrained features. Panel C confirms: zero-shot outperforms all finetuned variants. This is catastrophic forgetting on a small medical dataset."

---

## ✅ PRESENTATION CHECKLIST

- [ ] Load `fig4_method_overview.png` for methods slide
- [ ] Load `fig1_segmentation_comprehensive.png` for main results
- [ ] Load `fig2_clip_analysis.png` for CLIP results  
- [ ] Load `fig3_training_analysis.png` for training analysis
- [ ] Load `fig5_summary_results.png` for summary
- [ ] Memorize: 0.555, -33%, +64%, 44.4%
- [ ] Practice 6-minute timing (aim for 5:30)

---

## 📁 FILE LOCATIONS

```
results/figures/academic/
├── fig1_segmentation_comprehensive.png  ← MAIN RESULTS
├── fig2_clip_analysis.png               ← CLIP RESULTS
├── fig3_training_analysis.png           ← TRAINING ANALYSIS
├── fig4_method_overview.png             ← PIPELINE
└── fig5_summary_results.png             ← SUMMARY TABLES

All figures also available as PDF for high-quality printing.
```
