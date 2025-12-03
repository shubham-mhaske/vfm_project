# VFM Project: Complete Experiments & Results Summary

## Project Overview
**Goal**: Medical image segmentation on BCSS (Breast Cancer Semantic Segmentation) using SAM/SAM2 with CLIP-based classification.

## Dataset: BCSS
| Split | Images | Usage |
|-------|--------|-------|
| Train | 85 | SAM2 finetuning |
| Validation | 21 | Hyperparameter tuning |
| Test | 45 | Final evaluation |
| **Total** | **151** | |

**5 Tissue Classes**: Tumor, Stroma, Lymphocyte, Necrosis, Blood Vessel

---

## 📊 Main Results Summary

### Segmentation Performance (Dice Score)

| Model | Method | Dice | IoU | Notes |
|-------|--------|------|-----|-------|
| **SAM2 Hiera-L** | Box + Neg Points | **0.555** | 0.408 | Best overall |
| SAM2 Hiera-L | Bounding Box | 0.553 | 0.407 | Strong baseline |
| MedSAM ViT-B | Box + TTA | 0.536 | 0.390 | +1.4% from TTA |
| MedSAM ViT-B | Bounding Box | 0.522 | 0.378 | Baseline |
| SAM2 Hiera-L | Multi-point | 0.418 | 0.287 | 5 interior points |
| SAM2 Hiera-L | Centroid | 0.338 | 0.236 | Single point |

### Classification Performance (CLIP)

| Prompt Strategy | Accuracy | F1 Macro | Source |
|----------------|----------|----------|--------|
| **LLM Few-shot** | **44.4%** | 0.338 | GPT-4 |
| Hardcoded v2 | 42.2% | 0.312 | Manual |
| LLM CLIP-optimized | 35.6% | 0.270 | GPT-4 |
| LLM MM Few-shot | 29.4% | 0.220 | Gemini |
| Hardcoded v1 | 23.3% | 0.138 | Manual |

---

## 📈 Ablation Studies

### 1. Prompt Type Ablation (SAM2)
| Prompt | Dice | Δ vs Centroid |
|--------|------|---------------|
| Centroid | 0.338 | Baseline |
| Multi-Point (5) | 0.418 | +23.7% |
| Bounding Box | 0.553 | +63.6% |
| Box + Neg Points | 0.555 | **+64.2%** |

**Key Finding**: Bounding box prompts provide 64% improvement over single centroid points.

### 2. Test-Time Augmentation Ablation
| Model | Baseline | + Enhancement | Δ |
|-------|----------|---------------|---|
| MedSAM | 0.522 | 0.536 (TTA) | +1.4% |
| SAM2 | 0.553 | 0.555 (Neg) | +0.2% |

**Key Finding**: TTA provides marginal but consistent improvements.

### 3. Finetuning Ablation
| Config | Loss | Epochs | Test Dice | Δ vs ZS |
|--------|------|--------|-----------|---------|
| Zero-Shot | - | - | **0.555** | Baseline |
| SAM2 Focal | Focal+Dice | 50 | 0.372 | -33% |
| SAM2 LoRA | BCE+Dice | 30 | 0.355 | -36% |
| SAM2 Base | BCE+Dice | 100 | 0.371 | -33% |

**Key Finding**: Finetuning degraded performance, likely due to:
- Small dataset size (85 train images)
- Distribution shift between train/test
- Over-specialization to training set

### 4. CLIP Prompt Engineering
| Source | Strategy | Accuracy |
|--------|----------|----------|
| LLM | Text + Few-shot | **44.4%** |
| Manual | Hardcoded v2 | 42.2% |
| LLM | Text + CLIP-opt | 35.6% |
| LLM | Multimodal + Few-shot | 29.4% |

**Key Finding**: LLM-generated text prompts with few-shot examples outperform manual prompts by 2.2%.

---

## 🔬 Training Analysis

### Training Configurations
| Config | Parameters | Loss | LR | Epochs | Final Loss |
|--------|------------|------|-----|--------|------------|
| SAM2 Focal | 224M | Focal+Dice | 1e-4 | 50 | 29.0 |
| SAM2 LoRA | 4.2M | BCE+Dice | 1e-4 | 30 | 4.0 |
| SAM2 Base | 224M | BCE+Dice | 1e-4 | 100 | 1.72 |

### Training Dynamics
- **SAM2 Focal**: Focal loss drops 45→29, Dice loss 0.94→0.70
- **SAM2 LoRA**: Rapid convergence by epoch 7, total loss 36→4
- **SAM2 Base**: Gradual convergence, total loss 49→1.72

### Per-Class Performance (Validation)
| Class | Focal | LoRA | Base | Zero-Shot |
|-------|-------|------|------|-----------|
| Tumor | 0.386 | 0.320 | 0.432 | **0.560** |
| Stroma | 0.386 | 0.394 | 0.419 | **0.538** |
| Lymphocyte | 0.301 | 0.265 | 0.312 | **0.532** |
| Necrosis | 0.478 | 0.498 | 0.502 | **0.691** |
| Blood Vessel | 0.335 | 0.350 | 0.170 | **0.497** |

---

## 📁 Generated Figures

### Main Figures (from `scripts/generate_comparison_figures.py`)
| Figure | Description | Location |
|--------|-------------|----------|
| fig1 | SAM2 vs MedSAM bar chart | `results/figures/fig1_segmentation_comparison.*` |
| fig2 | Per-class heatmap | `results/figures/fig2_perclass_heatmap.*` |
| fig3 | CLIP comparison | `results/figures/fig3_clip_comparison.*` |
| fig4 | Prompt effect | `results/figures/fig4_prompt_effect.*` |
| fig5 | Radar comparison | `results/figures/fig5_radar_comparison.*` |

### Training & Ablation Figures (from `scripts/generate_training_analysis.py`)
| Figure | Description | Location |
|--------|-------------|----------|
| Training Curves | Loss over epochs | `results/figures/fig_training_curves.*` |
| ZS vs FT | Zero-shot vs Finetuned | `results/figures/fig_zeroshot_vs_finetuned.*` |
| Per-class FT | Per-class comparison | `results/figures/fig_perclass_finetuned.*` |
| Prompt Ablation | SAM2 prompt types | `results/figures/fig_ablation_prompts.*` |
| TTA Ablation | Test-time augmentation | `results/figures/fig_ablation_tta.*` |
| CLIP Ablation | Prompt engineering | `results/figures/fig_ablation_clip_prompts.*` |
| Dataset Stats | Class distribution | `results/figures/fig_dataset_stats.*` |
| Model Size | Parameters comparison | `results/figures/fig_model_size.*` |

### LaTeX Tables
| File | Contents |
|------|----------|
| `latex_tables.tex` | Main comparison tables |
| `ablation_tables.tex` | Ablation study tables |

---

## 📂 Directory Structure

```
results/
├── figures/                     # All publication figures
│   ├── fig1-5_*.png/pdf        # Main comparison figures
│   ├── fig_*.png/pdf           # Training & ablation figures
│   ├── latex_tables.tex        # Main LaTeX tables
│   └── ablation_tables.tex     # Ablation LaTeX tables
├── complete_metrics/           # Raw JSON metrics
│   ├── sam2_segmentation_*.json
│   ├── clip_classification_*.json
│   └── medsam_segmentation_*.json
├── exp*_*/                     # Individual experiment outputs
└── *_eval/                     # Finetuned model evaluations
```

---

## 🔑 Key Takeaways

### Segmentation
1. **SAM2 with bounding box prompts achieves best performance** (0.555 Dice)
2. **Zero-shot > Finetuned** for this dataset size and domain
3. **Prompt type matters more than model architecture** (64% improvement from centroid to box)
4. **TTA provides marginal improvements** (+1-2%)

### Classification
1. **LLM-generated prompts outperform manual prompts** (+2-21%)
2. **Few-shot examples improve prompt quality**
3. **Text-only prompts outperform multimodal prompts** for CLIP
4. **CLIP accuracy is limited** (max 44.4%) - consider domain-specific models

### Finetuning
1. **Small datasets hurt generalization** - finetuned models overfit
2. **LoRA reduces parameters but not overfitting**
3. **Loss function choice (Focal vs BCE) shows minimal impact**
4. **Need larger/more diverse training data for effective finetuning**

---

## 📋 Commands to Regenerate

```bash
# Generate comparison figures
python scripts/generate_comparison_figures.py

# Generate training & ablation figures  
python scripts/generate_training_analysis.py

# Run full evaluation
python src/evaluation.py \
  --sam_model_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --sam_checkpoint sam2/checkpoints/sam2.1_hiera_large.pt \
  --clip_prompts configs/prompts/hard_coded_prompts_v2.json \
  --output_dir results/new_experiment
```

---

*Last updated: Generated from experiment logs and results*
