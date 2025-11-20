# SAM2 BCSS Finetuning - Training Guide

## Training Configuration
- **Config**: `sam2/sam2/configs/bcss_finetune_best.yaml`
- **Epochs**: 20
- **Steps per epoch**: 500
- **Total training steps**: 10,000
- **Prompts**: box + negative points (explicit, from ground-truth masks)
- **Resolution**: 1024×1024
- **Batch size**: 1
- **Learning rate**: 2e-5 (base), cosine decay to 1e-6
- **Optimizer**: AdamW with weight decay 0.05

## Local Training (Mac CPU - for testing only)

```bash
conda activate vfm_project
cd /Users/shubhz/vfm_project

# 1-step smoke test (validates setup)
PYTHONPATH="$PWD:$PWD/sam2" python src/train_sam.py -c bcss_finetune_smoke

# Full training (use /tmp for logs to avoid disk space issues)
PYTHONPATH="$PWD:$PWD/sam2" python src/train_sam.py -c bcss_finetune_best \
  launcher.experiment_log_dir=/tmp/finetune_logs_best
```

## TAMU Grace HPRC Training (GPU)

```bash
# On login node, navigate to project
cd $SCRATCH/vfm_project

# Create logs directory
mkdir -p logs

# Submit job
sbatch run_gpu.slurm

# Monitor training (replace JOBID with actual job ID)
tail -f logs/sam_finetune_JOBID.out

# Check job status
squeue -u $USER
```

**Slurm config** (`run_gpu.slurm`):
- GPU: A100 (1×)
- CPUs: 16
- Memory: 64GB
- Runtime: 2 days
- Partition: gpu

## Evaluation

After training completes:

```bash
conda activate vfm_project
cd /Users/shubhz/vfm_project

python src/evaluate_segmentation.py \
  --checkpoint finetune_logs/checkpoints/checkpoint.pt \
  --model_cfg sam2/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml \
  --image_dir data/bcss/images \
  --mask_dir data/bcss/masks \
  --split val \
  --prompt_type box \
  --use_neg_points
```

## Key Files

| File | Purpose |
|------|---------|
| `sam2/sam2/configs/bcss_finetune_best.yaml` | Main training config (GPU-optimized) |
| `sam2/sam2/configs/bcss_finetune_smoke.yaml` | 1-epoch smoke test config |
| `run_gpu.slurm` | TAMU Grace HPRC submission script |
| `src/train_sam.py` | Training wrapper (sets PYTHONPATH) |
| `src/finetune_dataset.py` | BCSS dataset with explicit prompt generation |
| `src/evaluate_segmentation.py` | Evaluation script |

## Dataset

- **Train**: 85 images (BCSS split)
- **Validation**: Remaining images in BCSS
- **Prompts**: Auto-generated per object (centroid, box, multi-point)
- **Path**: `data/bcss/images/` and `data/bcss/masks/`

## Expected Training Time

- **Grace A100**: ~1-2 hours for 20 epochs
- **Mac CPU**: ~30+ minutes per epoch (not recommended)

## Checkpoints

- **Location**: `finetune_logs/checkpoints/checkpoint.pt`
- **Size**: ~200MB per save
- **Save frequency**: Every epoch (1 per epoch)

## Notes

- Explicit prompts are generated from ground-truth masks during training
- Evaluation uses same prompt type and settings as training for consistency
- Non-strict checkpoint loading allows architecture flexibility
- AMP (Automatic Mixed Precision) enabled for GPU to reduce memory and speed up training

## 📊 **Quick Experimental Plan (Improved)**

### **Step 1: Evaluate Existing Checkpoints**
Before starting new training, check if your long training run (epoch 400+) actually improved performance.
```bash
python src/evaluate_checkpoint.py \
  --checkpoint finetune_logs/checkpoints/checkpoint_439.pt \
  --config sam2.1_hiera_b+.yaml \
  --output_dir results/eval_checkpoint_439
```

### **Step 2: Run Improved Training (Recommended)**
We have created a new configuration `configs/bcss_finetune_improved.yaml` that includes:
- **Larger Model:** `hiera_l` (Large) instead of `hiera_b+`
- **Stronger Augmentation:** Color jitter, rotation, vertical flips
- **Mixed Prompts:** Randomly switches between box and centroid prompts
- **Optimized Hyperparameters:** Adjusted learning rate and phases

**To run this experiment:**
```bash
# 1. Download the large checkpoint if you haven't
bash sam2/checkpoints/download_ckpts.sh

# 2. Start training
python src/train_sam.py -c bcss_finetune_improved
```

### **Step 3: Compare Results**
After training, evaluate the new model:
```bash
python src/evaluate_checkpoint.py \
  --checkpoint finetune_logs_improved/checkpoints/checkpoint_49.pt \
  --config sam2.1_hiera_l.yaml \
  --output_dir results/eval_improved_model
```

## 🎯 **Expected Improvements**
- **Dice:** 0.43 → **0.55-0.65** (matching/exceeding baseline)
- **IoU:** 0.28 → **0.40-0.50**
