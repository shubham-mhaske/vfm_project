#!/bin/bash
#SBATCH --job-name=sam2_metrics
#SBATCH --output=slurm_logs/sam2_metrics_%j.out
#SBATCH --error=slurm_logs/sam2_metrics_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00

# SAM2 Segmentation Metrics Collection
# Evaluates 6 prompt configurations on test set

echo "=========================================="
echo "SAM2 SEGMENTATION METRICS"
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "=========================================="

# Environment setup
module restore dl
source $SCRATCH/vfm_env/bin/activate

# Project paths
PROJECT_ROOT="/scratch/user/$(whoami)/vfm_project"
cd $PROJECT_ROOT

export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src:$PROJECT_ROOT/sam2:$PYTHONPATH"
export OMP_NUM_THREADS=24
export MKL_NUM_THREADS=24

# Create output directory
mkdir -p results/complete_metrics
mkdir -p slurm_logs

echo "Running SAM2 evaluation..."
python -c "
import os
import sys
import json
import time
from datetime import datetime

sys.path.insert(0, '$PROJECT_ROOT')
sys.path.insert(0, '$PROJECT_ROOT/src')
sys.path.insert(0, '$PROJECT_ROOT/sam2')

import numpy as np
import torch
from tqdm import tqdm

from dataset import BCSSDataset
from sam_segmentation import (
    get_sam2_predictor, get_prompts_from_mask,
    get_predicted_mask_from_prompts, calculate_metrics
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

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

# Load SAM2
print('Loading SAM2...')
os.chdir('$PROJECT_ROOT/sam2')
try:
    from training.utils.train_utils import register_omegaconf_resolvers
    register_omegaconf_resolvers()
except: pass

predictor = get_sam2_predictor(
    'configs/sam2.1/sam2.1_hiera_l.yaml',
    '$PROJECT_ROOT/sam2/checkpoints/sam2.1_hiera_large.pt',
    device
)
os.chdir('$PROJECT_ROOT')
print('SAM2 loaded!')

# Prompt configurations
prompt_configs = [
    {'name': 'centroid', 'prompt_type': 'centroid', 'use_neg_points': False},
    {'name': 'multi_point', 'prompt_type': 'multi_point', 'use_neg_points': False},
    {'name': 'box_baseline', 'prompt_type': 'box', 'use_neg_points': False},
    {'name': 'box_neg_points', 'prompt_type': 'box', 'use_neg_points': True},
]

all_results = {}
start_time = time.time()

for config in prompt_configs:
    config_name = config['name']
    print(f'\n--- Evaluating: {config_name} ---')
    
    class_metrics = {cid: {'dice': [], 'iou': []} for cid in target_class_ids}
    
    for i in tqdm(range(len(dataset)), desc=config_name):
        sample = dataset[i]
        image = sample['image_np']
        gt_mask = sample['mask'].numpy()
        
        predictor.set_image(image)
        
        for class_id in np.unique(gt_mask):
            if class_id == 0 or class_id not in class_names:
                continue
            
            binary_gt = (gt_mask == class_id).astype(np.uint8)
            if binary_gt.sum() == 0:
                continue
            
            prompts = get_prompts_from_mask(binary_gt)
            if config['prompt_type'] not in prompts:
                continue
            
            pred_mask, _, _ = get_predicted_mask_from_prompts(
                predictor, image, prompts,
                prompt_type=config['prompt_type'],
                use_neg_points=config.get('use_neg_points', False)
            )
            pred_mask = (pred_mask > 0.5).astype(np.uint8)
            
            dice, iou = calculate_metrics(pred_mask, binary_gt)
            class_metrics[class_id]['dice'].append(dice)
            class_metrics[class_id]['iou'].append(iou)
    
    # Aggregate
    results = {'config': config, 'per_class': {}, 'overall': {}}
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
    
    all_results[config_name] = results
    print(f'  Dice: {results[\"overall\"][\"dice_mean\"]:.4f}, IoU: {results[\"overall\"][\"iou_mean\"]:.4f}')

elapsed = time.time() - start_time
print(f'\nCompleted in {elapsed/60:.1f} minutes')

# Save results
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_path = f'$PROJECT_ROOT/results/complete_metrics/sam2_segmentation_{timestamp}.json'
with open(output_path, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f'Results saved to: {output_path}')
"

echo ""
echo "=========================================="
echo "SAM2 metrics collection complete!"
echo "Finished: $(date)"
echo "=========================================="
