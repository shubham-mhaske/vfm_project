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

# Download SAM2 checkpoints
bash sam2/checkpoints/download_ckpts.sh
```

### 2. Training

```bash
# Local test (1 epoch)
python src/run_finetuning.py experiment=base_finetune_v2_stable scratch.num_epochs=1

# Full training
python src/run_finetuning.py experiment=base_finetune_v2_stable

# On HPRC Grace
sbatch scripts/slurm/run_training_v3.slurm
```

### 3. Evaluation

```bash
python src/evaluation.py \
  --sam_checkpoint finetune_logs/base_finetune_v2_stable-*/checkpoints/checkpoint_15.pt \
  --sam_model_cfg sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml \
  --clip_prompts configs/prompts/hard_coded_prompts_v2.json \
  --output_dir results/my_evaluation
```

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

**Note**: Blood vessel is class ID **18** (not 5). Class 5 in BCSS is glandular_secretions.

### Data Split (Deterministic)

- **Train**: 85 images
- **Validation**: 21 images  
- **Test**: 45 images (TCGA prefixes: OL-, LL-, E2-, EW-, GM-, S3-)

## Current Results

### Best Model: v2_stable (Epoch 15)

| Class | Dice Score |
|-------|------------|
| Tumor | 0.54 |
| Stroma | 0.43 |
| Lymphocyte | 0.20 |
| Necrosis | 0.45 |
| Blood Vessel | 0.03 |
| **Overall** | **0.42** |

See `EXPERIMENTS.md` for full training history and analysis.

## HPC Setup

For TAMU HPRC Grace cluster setup, see `docs/HPRC_GRACE_SETUP.md`.

## License

MIT License - See LICENSE file.
