# 6-Minute Academic Presentation Guide

## 📊 PUBLICATION-QUALITY FIGURES

### Qualitative Results (Real Test Images) ⭐ NEW
Location: `results/figures/qualitative/`

| Figure | Description | Recommended Slide |
|--------|-------------|-------------------|
| **qualitative_method_comparison.png** | 4 test images × 6 methods side-by-side | Slide 5-6: Main Results ⭐⭐ |
| **qualitative_per_class.png** | Per-tissue class segmentation examples | Slide 7: Per-Class Analysis |
| **qualitative_prompt_comparison.png** | Prompt strategy visual comparison | Slide 4: Prompt Ablation |
| **qualitative_success_failure.png** | Success (Dice>0.7) vs Failure (Dice<0.4) | Slide 10: Limitations |
| **qualitative_full_segmentation.png** | Full multi-class colored segmentation | Slide 3: Pipeline Demo |

### Academic Charts (Metrics-Based)
Location: `results/figures/academic/`

| Figure | Description | Recommended Slide |
|--------|-------------|-------------------|
| **fig4_method_overview.png** | Pipeline architecture schematic | Slide 3: Methods |
| **fig1_segmentation_comprehensive.png** | 4-panel segmentation analysis | Slide 8: Quantitative Results |
| **fig2_clip_analysis.png** | 3-panel CLIP classification | Slide 9: CLIP Results |
| **fig3_training_analysis.png** | 3-panel training curves | Slide 10: Why Finetuning Failed |
| **fig5_summary_results.png** | Complete results tables | Slide 11: Summary |

---

## ⏱️ 12-SLIDE STRUCTURE (6 minutes)

| Slide | Time | Figure | Key Message |
|-------|------|--------|-------------|
| 1. Title | 10s | Sample image | VFMs for Medical Image Segmentation |
| 2. Problem | 30s | - | Manual annotation costly; VFMs pretrained on billions |
| 3. Methods | 40s | `academic/fig4_method_overview.png` | Two-stage: SAM2→CLIP pipeline |
| 4. Dataset | 20s | - | BCSS: 151 images, 5 classes, 45 test |
| **5-6. Segmentation** | **60s** | **`qualitative/qualitative_method_comparison.png`** | **Real predictions! Box+Neg best (0.555)** |
| **7. Per-Class** | **30s** | **`qualitative/qualitative_per_class.png`** | **Visual per-class performance** |
| **8. CLIP** | **40s** | **`academic/fig2_clip_analysis.png`** | **Few-shot LLM best (44.4%)** |
| **9. Training** | **40s** | **`academic/fig3_training_analysis.png`** | **Overfitting evidence** |
| 10. Limitations | 20s | `qualitative/qualitative_success_failure.png` | Success vs failure cases |
| 11. Summary | 30s | `academic/fig5_summary_results.png` | All results in tables |
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
results/figures/
├── qualitative/                              ← REAL TEST IMAGES ⭐
│   ├── qualitative_method_comparison.png     ← MAIN PRESENTATION FIGURE
│   ├── qualitative_per_class.png             ← Per-class examples
│   ├── qualitative_prompt_comparison.png     ← Prompt ablation visual
│   ├── qualitative_success_failure.png       ← Success vs failure
│   └── qualitative_full_segmentation.png     ← Multi-class demo
│
├── academic/                                 ← METRICS-BASED CHARTS
│   ├── fig1_segmentation_comprehensive.png   ← 4-panel quantitative
│   ├── fig2_clip_analysis.png                ← CLIP results
│   ├── fig3_training_analysis.png            ← Training curves
│   ├── fig4_method_overview.png              ← Pipeline schematic
│   └── fig5_summary_results.png              ← Summary tables
│
└── All figures also available as PDF for high-quality printing.
```

### Generation Scripts

```bash
# Generate qualitative figures (requires GPU + SAM2/MedSAM)
sbatch scripts/slurm/run_qualitative_figures.slurm

# Generate academic charts (CPU only, from metrics)
python scripts/analysis/generate_academic_figures.py
```
