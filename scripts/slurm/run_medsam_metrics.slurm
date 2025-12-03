#!/bin/bash
#SBATCH --job-name=medsam_metrics
#SBATCH --output=slurm_logs/medsam_metrics_%j.out
#SBATCH --error=slurm_logs/medsam_metrics_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:30:00

# MedSAM Segmentation Metrics Collection
# Evaluates with and without TTA

echo "=========================================="
echo "MEDSAM SEGMENTATION METRICS"
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "=========================================="

# Environment setup
module restore dl
source $SCRATCH/vfm_env/bin/activate

# Project paths
PROJECT_ROOT="/scratch/user/$(whoami)/vfm_project"
cd $PROJECT_ROOT

export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src:$PROJECT_ROOT/sam2:$PROJECT_ROOT/MedSAM:$PYTHONPATH"
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

# Create output directory
mkdir -p results/complete_metrics
mkdir -p slurm_logs

echo "Running MedSAM evaluation..."
python -c "
import os
import sys
import json
import time
from datetime import datetime

sys.path.insert(0, '$PROJECT_ROOT')
sys.path.insert(0, '$PROJECT_ROOT/src')
sys.path.insert(0, '$PROJECT_ROOT/MedSAM')

import numpy as np
import torch
from tqdm import tqdm

from dataset import BCSSDataset
from sam_segmentation import get_prompts_from_mask, calculate_metrics
from evaluate_medsam import (
    load_medsam_model,
    predict_with_medsam,
    predict_with_medsam_tta
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')
if device == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')

# Class mappings
class_names = {0: 'background', 1: 'tumor', 2: 'stroma', 3: 'lymphocyte', 4: 'necrosis', 18: 'blood_vessel'}
target_class_ids = [1, 2, 3, 4, 18]

# Load dataset
print('Loading dataset...')
dataset = BCSSDataset(
    image_dir='$PROJECT_ROOT/data/bcss/images',
    mask_dir='$PROJECT_ROOT/data/bcss/masks',
    split='test'
)
print(f'Loaded {len(dataset)} test samples')

# Load MedSAM using the proper function from src/evaluate_medsam.py
print('Loading MedSAM...')
checkpoint_path = '$PROJECT_ROOT/models/medsam_checkpoints/medsam_vit_b.pth'
medsam_model = load_medsam_model(checkpoint_path, device)

# Evaluate both with and without TTA
configs = [
    {'name': 'medsam_box', 'use_tta': False},
    {'name': 'medsam_box_tta', 'use_tta': True},
]

all_results = {}
start_time = time.time()

for config in configs:
    config_name = config['name']
    use_tta = config['use_tta']
    
    print(f'\n--- Evaluating: {config_name} (TTA={use_tta}) ---')
    
    class_metrics = {cid: {'dice': [], 'iou': []} for cid in target_class_ids}
    
    for i in tqdm(range(len(dataset)), desc=config_name):
        sample = dataset[i]
        image = sample['image_np']
        gt_mask = sample['mask'].numpy()
        
        for class_id in np.unique(gt_mask):
            if class_id == 0 or class_id not in class_names:
                continue
            
            binary_gt = (gt_mask == class_id).astype(np.uint8)
            if binary_gt.sum() == 0:
                continue
            
            prompts = get_prompts_from_mask(binary_gt)
            if 'box' not in prompts:
                continue
            
            box = prompts['box']
            
            try:
                if use_tta:
                    pred_mask, _ = predict_with_medsam_tta(medsam_model, image, box, device)
                else:
                    pred_mask, _ = predict_with_medsam(medsam_model, image, box, device)
                
                dice, iou = calculate_metrics(pred_mask, binary_gt)
                class_metrics[class_id]['dice'].append(dice)
                class_metrics[class_id]['iou'].append(iou)
            except Exception as e:
                print(f'  Error processing sample {i}, class {class_id}: {e}')
                continue
    
    # Aggregate
    results = {'per_class': {}, 'overall': {}}
    all_dice, all_iou = [], []
    
    for cid in target_class_ids:
        cname = class_names[cid]
        dice_scores = class_metrics[cid]['dice']
        iou_scores = class_metrics[cid]['iou']
        
        if dice_scores:
            results['per_class'][cname] = {
                'dice_mean': float(np.mean(dice_scores)),
                'dice_std': float(np.std(dice_scores)),
                'iou_mean': float(np.mean(iou_scores)),
                'iou_std': float(np.std(iou_scores)),
                'count': len(dice_scores)
            }
            all_dice.extend(dice_scores)
            all_iou.extend(iou_scores)
    
    results['overall'] = {
        'dice_mean': float(np.mean(all_dice)) if all_dice else 0,
        'dice_std': float(np.std(all_dice)) if all_dice else 0,
        'iou_mean': float(np.mean(all_iou)) if all_iou else 0,
        'iou_std': float(np.std(all_iou)) if all_iou else 0,
        'total_samples': len(all_dice)
    }
    results['use_tta'] = use_tta
    
    all_results[config_name] = results
    print(f'  Dice: {results[\"overall\"][\"dice_mean\"]:.4f}, IoU: {results[\"overall\"][\"iou_mean\"]:.4f}')

elapsed = time.time() - start_time
print(f'\nCompleted in {elapsed/60:.1f} minutes')

# Save results
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_path = f'$PROJECT_ROOT/results/complete_metrics/medsam_segmentation_{timestamp}.json'
with open(output_path, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f'Results saved to: {output_path}')
"

echo ""
echo "=========================================="
echo "MedSAM metrics collection complete!"
echo "Finished: $(date)"
echo "=========================================="
