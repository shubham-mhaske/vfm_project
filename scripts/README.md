# Scripts Directory

Organized utilities for SAM2 histopathology segmentation training and evaluation.

## Directory Structure

```
scripts/
├── slurm/              # SLURM job scripts
│   ├── run_training_v3.slurm
│   ├── run_training_grace.slurm
│   └── evaluate_segmentation.slurm
├── validation/         # Validation and evaluation tools
│   └── validate_perclass.py
├── training/          # Training utilities
│   └── test_v3_config.sh
├── legacy/            # Old scripts (archived)
│   └── test_training_local.sh
├── validate_config.py # Config validation
└── cleanup.sh         # Cleanup duplicate results
```

## Validation Scripts

### `validation/validate_perclass.py`

Per-class validation for SAM2 checkpoints. Evaluates each tissue class separately (realistic metric).

**Usage:**
```bash
python scripts/validation/validate_perclass.py \
  --checkpoint finetune_logs/base_finetune_v3_*/checkpoints/checkpoint_15.pt \
  --model_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --split val \
  --num_samples 5  # Optional: limit samples for quick test
```

**Arguments:**
- `--checkpoint`: Path to SAM2 checkpoint (.pt file)
- `--model_cfg`: SAM2 config (relative to sam2 package, e.g., `configs/sam2.1/sam2.1_hiera_l.yaml`)
- `--split`: Dataset split (`val` or `test`)
- `--num_samples`: Optional limit for quick testing
- `--output`: Optional JSON output file for results

**Output:**
```
Per-Class Results:
tumor          : Dice 0.5917 ± 0.1818 (3 samples)
stroma         : Dice 0.3252 ± 0.0351 (3 samples)
lymphocyte     : Dice 0.3946 ± 0.3054 (3 samples)
necrosis       : Dice 0.4220 ± 0.2819 (3 samples)
blood_vessel   : No samples
Overall Per-Class Dice: 0.4334
```

## Training Scripts

### `training/test_v3_config.sh`

Local testing script for v3 config before HPRC submission.

**Usage:**
```bash
bash scripts/training/test_v3_config.sh
```

**What it does:**
1. Validates config syntax
2. Runs dry run (0 epochs)
3. Tests single epoch with 2 samples

**Expected time:** 5-10 minutes

## Cleanup Scripts

### `cleanup.sh`

Removes duplicate and failed evaluation runs.

**Usage:**
```bash
bash scripts/cleanup.sh
```

**What it removes:**
- Duplicate zero-shot evaluations
- Failed/incomplete runs
- Temporary files and pycache

**What it keeps:**
- Best checkpoints
- Final evaluation results
- Original experiment data

## Common Workflows

### 1. Before Training (Local Test)

```bash
# Test v3 config locally
bash scripts/training/test_v3_config.sh

# If successful, submit to HPRC
sbatch scripts/slurm/run_training_v3.slurm
```

### 2. During Training (Monitor Progress)

```bash
# SSH to HPRC
ssh grace1

# Check training progress
tail -f finetune_logs/v3_perclass-*.out

# Monitor GPU usage
watch nvidia-smi

# View tensorboard
tensorboard --logdir=finetune_logs/base_finetune_v3_perclass-*/tensorboard --port 6006
```

### 3. After Training (Validation)

```bash
# Quick validation on val set (3 samples)
python scripts/validation/validate_perclass.py \
  --checkpoint finetune_logs/base_finetune_v3_*/checkpoints/checkpoint_15.pt \
  --model_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --split val \
  --num_samples 3

# Full validation (all val samples)
python scripts/validation/validate_perclass.py \
  --checkpoint finetune_logs/base_finetune_v3_*/checkpoints/checkpoint_15.pt \
  --model_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --split val \
  --output results/v3_validation.json

# Full evaluation with CLIP
python src/evaluation.py \
  --sam_checkpoint finetune_logs/base_finetune_v3_*/checkpoints/checkpoint_15.pt \
  --sam_model_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --clip_prompts configs/prompts/hard_coded_prompts_v2.json \
  --output_dir results/v3_evaluation
```

### 4. Cleanup (Remove Duplicates)

```bash
# Remove duplicate results
bash scripts/cleanup.sh

# Check what will be deleted (dry run)
bash scripts/cleanup.sh --dry-run  # Add this flag if implementing
```

## Tips

1. **Always test locally first**: Run `scripts/training/test_v3_config.sh` before HPRC submission
2. **Monitor early**: Check first epoch output to catch errors early
3. **Save checkpoints frequently**: v3 config saves every 5 epochs
4. **Use per-class validation**: More accurate than training's foreground detection metric
5. **Compare with baselines**: v2 achieved Dice 0.42, v3 targets 0.60-0.70

## Troubleshooting

### "Cannot find primary config" error
- **Issue**: Model config path is wrong
- **Fix**: Use relative path like `configs/sam2.1/sam2.1_hiera_l.yaml` (not absolute path)

### "AttributeError: 'Tensor' object has no attribute 'astype'"
- **Issue**: Mask is tensor, not numpy array
- **Fix**: Already handled in validate_perclass.py with `torch.is_tensor()` check

### "KeyError: 7" in validation
- **Issue**: Dataset has class IDs not in class_names mapping
- **Fix**: Already handled - script skips unknown classes

### Training crashes at epoch 0
- **Issue**: Usually OOM or data loading problem
- **Fix**: Reduce batch size or check data paths

## File Locations

- **Configs**: `conf/experiment/base_finetune_v3_perclass.yaml`
- **Checkpoints**: `finetune_logs/base_finetune_v3_perclass-*/checkpoints/`
- **Logs**: `finetune_logs/v3_perclass-*.out`
- **Results**: `results/v3_evaluation/`
- **Master Tracker**: `EXPERIMENTS.md`
