# Promptable Pathology: Interactive Zero-Shot Pipeline for Classifying Tumor Microenvironments

**CSCE 689 - Visual Foundation Models for Medical Image Analysis (Fall 2025)**

A comprehensive framework for zero-shot histopathology analysis using Visual Foundation Models (SAM2 + CLIP). This project demonstrates that prompt engineering outperforms fine-tuning for small medical datasets.

## Key Results

| Component | Best Method | Score |
|-----------|-------------|-------|
| **Segmentation** | SAM2 + Box + Neg + TTA | **0.550 Dice** |
| **Classification** | CLIP + LogReg Classifier | **40.4% Accuracy** |
| **Prompt Engineering (SAM2)** | Box + Neg + TTA | +4.6% over baseline |
| **Prompt Engineering (CLIP)** | Fewshot prompts | +13.2% over baseline |

## Features

- **Zero-Shot Segmentation**: SAM2 with optimized prompting strategies (box, negative points, TTA)
- **Zero-Shot Classification**: CLIP with prompt engineering and feature-based classifiers
- **LLM-Powered Prompts**: Auto-generated text and multimodal prompts via Gemini
- **Comprehensive Evaluation**: Per-class Dice/IoU/Accuracy metrics and confusion matrices
- **Alternative Models**: PLIP (pathology CLIP), MedSAM, ensemble methods
- **HPC Ready**: SLURM scripts for TAMU HPRC Grace cluster

## Directory Structure

```
vfm_project/
├── conf/                         # Hydra configuration
│   ├── config.yaml
│   └── experiment/               # Training experiment configs
├── configs/prompts/              # CLIP prompt JSON files
├── data/bcss/                    # Dataset (images + masks)
├── docs/                         # Documentation
│   ├── FINAL_RESULTS_SUMMARY.md  # Complete results overview
│   ├── EXPERIMENT_COMPARISON.md  # Comparison framework
│   └── HPRC_GRACE_SETUP.md       # HPC setup guide
├── finetune_logs/                # Training outputs
│   ├── sam2_base_100ep/          # Base finetuning (100 epochs)
│   ├── sam2_focal_50ep/          # Focal loss (50 epochs)
│   ├── sam2_lora_30ep/           # LoRA adapter (30 epochs)
│   └── pathsam2_ctranspath/      # CTransPath encoder
├── results/                      # Evaluation results
│   ├── figures/                  # Publication figures (PNG/PDF)
│   ├── complete_metrics/         # Raw JSON metrics
│   ├── sam2_zeroshot_hardcoded/  # Best SAM2 results
│   ├── medsam_box_tta/           # MedSAM with TTA
│   ├── clip_llm_text_fewshot/    # Best CLIP results
│   └── sam2_prompt_ablation/     # Prompt type ablation
├── scripts/                      # Utility scripts
│   ├── analysis/                 # Figure/table generation
│   ├── slurm/                    # HPC job scripts
│   ├── training/                 # Training launchers
│   └── utils/                    # Download helpers
├── src/                          # Source code
│   ├── dataset.py                # BCSS data loader
│   ├── sam_segmentation.py       # SAM2 segmentation
│   ├── clip_classification.py    # CLIP classification
│   ├── evaluation.py             # Main evaluation pipeline
│   ├── evaluators/               # Evaluation scripts
│   ├── trainers/                 # Training scripts
│   └── prompt_generators/        # LLM prompt generation
├── sam2/                         # SAM2 submodule
├── MedSAM/                       # MedSAM submodule
└── models/                       # Model checkpoints
```

## Quick Start

### 1. Setup
```bash
# Clone with submodules
git clone <repo-url> && cd vfm_project
git submodule update --init --recursive

# Install dependencies
pip install -r requirements.txt

# Download pre-trained weights
bash sam2/checkpoints/download_ckpts.sh
```

### 2. Run Best Configuration (Zero-Shot)
```bash
# SAM2 Segmentation + CLIP Classification
python src/evaluation.py \
  --sam_model_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --sam_checkpoint sam2/checkpoints/sam2.1_hiera_large.pt \
  --clip_prompts configs/prompts/llm_text_prompts_v3_fewshot.json \
  --output_dir results/best_zeroshot

# CLIP Feature Classifier (best classification: 40.4%)
python src/train_clip_classifier.py \
  --output_dir results/clip_classifier
```

### 3. Generate Publication Figures
```bash
# Generate comparison figures
python scripts/analysis/generate_comparison_figures.py

# Generate training analysis & ablation studies
python scripts/analysis/generate_training_analysis.py

# Outputs: results/figures/ (13 PNG/PDF figures + LaTeX tables)
```

## Recent Experiments (December 2025)

### Classification Experiments
| Method | Accuracy | Notes |
|--------|----------|-------|
| CLIP Hardcoded v1 | 25.7% | Basic prompts |
| CLIP Hardcoded v2 | 35.6% | Improved prompts |
| LLM Text Fewshot | **38.9%** | Best zero-shot |
| CLIP + LogReg | **40.4%** | Best overall |
| PLIP Zero-shot | 26.9% | Medical CLIP - worse |
| CLIP + PLIP Ensemble | 31.4% | Ensemble hurt |
| Multi-scale CLIP | 27.8% | More context hurt |

### Segmentation Experiments
| Method | Dice | Notes |
|--------|------|-------|
| SAM2 Box + Neg + TTA | **0.550** | Best |
| SAM2 Box baseline | 0.526 | Good |
| MedSAM Box + TTA | 0.536 | 5% worse |
| Fine-tuned SAM2 | 0.320 | Failed |

### Key Finding: Fine-tuning Failed
All fine-tuning approaches performed **worse** than zero-shot:
- **Root cause**: Only 85 training images → catastrophic forgetting
- **Lesson**: For small datasets, invest in prompt engineering, not fine-tuning

## Dataset

**BCSS (Breast Cancer Semantic Segmentation)** - 151 H&E stained tissue patches (1024×1024)

### Class Distribution
| Class | ID | Frequency |
|-------|-----|-----------|
| Tumor | 1 | 33.8% |
| Stroma | 2 | 26.2% |
| Necrosis | 4 | 6.9% |
| Lymphocyte | 3 | 5.9% |
| Blood Vessel | 18 | 0.5% |

### Data Split
- **Train**: 85 images
- **Validation**: 21 images  
- **Test**: 45 images

## Per-Class Results

### Classification (CLIP + LogReg)
| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Tumor | 37.1% | 28.9% | 32.5% |
| Stroma | 43.1% | 55.6% | 48.5% |
| Lymphocyte | 39.6% | 51.4% | 44.7% |
| Necrosis | 42.9% | 26.1% | 32.4% |
| Blood Vessel | 0.0% | 0.0% | 0.0% |

### Segmentation (SAM2 + Box + TTA)
| Class | Dice |
|-------|------|
| Necrosis | 0.708 |
| Tumor | 0.565 |
| Lymphocyte | 0.562 |
| Stroma | 0.540 |
| Blood Vessel | 0.504 |

## HPC Setup

For TAMU HPRC Grace cluster setup, see `docs/HPRC_GRACE_SETUP.md`.

## Generated Figures

Run figure generation scripts to create publication-ready outputs:

### Main Figures (`results/figures/`)
- `fig1_segmentation_comparison.*` - SAM2 vs MedSAM performance
- `fig2_perclass_heatmap.*` - Per-class Dice heatmap
- `fig3_clip_comparison.*` - CLIP prompt strategies
- `fig4_prompt_effect.*` - SAM2 prompt type impact
- `fig5_radar_comparison.*` - Multi-metric radar chart

### Ablation Study Figures
- `fig_training_curves.*` - Training loss over epochs
- `fig_zeroshot_vs_finetuned.*` - Zero-shot vs finetuned
- `fig_ablation_prompts.*` - Prompt type ablation
- `fig_ablation_tta.*` - TTA effect analysis
- `fig_ablation_clip_prompts.*` - CLIP prompt engineering

### LaTeX Tables
- `latex_tables.tex` - Main comparison tables
- `ablation_tables.tex` - Ablation study tables

## License

MIT License - See LICENSE file.

---
**Last Updated**: December 2, 2025
