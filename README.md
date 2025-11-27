# Visual Foundation Model for Histopathology Image Segmentation

Framework for utilizing and evaluating Visual Foundation Models (SAM/SAM2) for medical image segmentation on the BCSS (Breast Cancer Semantic Segmentation) dataset.

## Features

- **Zero-Shot Segmentation**: Pre-trained SAM with various prompting strategies
- **LLM-Powered Prompts**: Auto-generated text and multimodal prompts via Gemini
- **Fine-Tuning**: SAM2 fine-tuning on BCSS with Hydra configs
- **Comprehensive Evaluation**: Per-class Dice/IoU metrics and confusion matrices
- **HPC Ready**: SLURM scripts for TAMU HPRC Grace cluster

## Directory Structure

```
├── conf/                     # Hydra configuration
│   └── experiment/           # Training configs (v2_stable, v3_perclass)
├── configs/prompts/          # CLIP prompt files
├── data/bcss/                # Dataset (images + masks)
├── docs/                     # Setup guides
├── experiments/              # Experiment tracking
├── finetune_logs/            # Training outputs & checkpoints
├── results/                  # Evaluation results
├── sam2/                     # SAM2 submodule
├── scripts/                  # Utility scripts
│   ├── slurm/               # HPC job scripts
│   ├── training/            # Local training scripts
│   └── validation/          # Validation utilities
├── src/                      # Source code
│   ├── run_finetuning.py    # Main training script
│   ├── evaluation.py        # Full evaluation pipeline
│   ├── dataset.py           # BCSS Dataset loader
│   ├── finetune_dataset.py  # Training dataset
│   └── sam_segmentation.py  # SAM utilities
└── EXPERIMENTS.md            # Experiment tracker
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
python src/download_ctranspath_weights.py
```

### 2. Training
```bash
# Run the optimized Path-SAM2 + CTransPath training
sbatch scripts/slurm/run_path_sam2_ctranspath_optimized.slurm

# Or locally:
python src/run_path_sam2_ctranspath.py experiment=path_sam2_ctranspath_optimized
```

### 3. Evaluation
```bash
# Run the full evaluation pipeline on a completed training run
bash scripts/evaluation/evaluate_experiment.sh finetune_logs/path_sam2_focal-2025-11-27_12-00-00
```

## Recent Improvements (November 2025)

- **Path-SAM2 + CTransPath**: Integrated a pre-trained CTransPath encoder for histopathology-specific features, significantly boosting performance over the baseline.
- **Optimized Training**: Achieved a ~5-8x speedup with `bfloat16` AMP, increased batch sizes, and optimized data loaders.
- **Focal Loss**: Implemented a `FocalDiceLoss` to address severe class imbalance, dramatically improving results for rare classes like `blood_vessel`.
- **Stain Normalization**: Added on-the-fly Macenko stain normalization and augmentation to handle variations in H&E staining.
- **Test-Time Augmentation (TTA)**: Integrated TTA (flips and rotations) at inference time for more robust predictions.
- **Automated Evaluation**: Created a comprehensive evaluation pipeline to find the best checkpoint, optimize thresholds, and run final evaluation.

## Dataset

**BCSS (Breast Cancer Semantic Segmentation)** - 151 H&E stained tissue patches (1024×1024)

### Class Mapping
| Class ID | Tissue Type | Frequency |
|----------|-------------|-----------|
| 0 | Background | - |
| 1 | Tumor | 33.8% |
| 2 | Stroma | 26.2% |
| 3 | Lymphocyte | 5.9% |
| 4 | Necrosis | 6.9% |
| 18 | Blood Vessel | 0.5% |

**Note**: Blood vessel is class ID **18** (not 5).

### Data Split (Deterministic)
- **Train**: 85 images
- **Validation**: 21 images  
- **Test**: 45 images

## Expected Results (with all improvements)

| Class | Dice Score |
|-------|------------|
| Tumor | 0.68 - 0.72 |
| Stroma | 0.62 - 0.66 |
| Lymphocyte | 0.38 - 0.45 |
| Necrosis | 0.62 - 0.68 |
| Blood Vessel | 0.25 - 0.30 |
| **Overall** | **0.70 - 0.75** |

See `EXPERIMENTS.md` for a full breakdown of each improvement's contribution.


## HPC Setup

For TAMU HPRC Grace cluster setup, see `docs/HPRC_GRACE_SETUP.md`.

## License

MIT License - See LICENSE file.
