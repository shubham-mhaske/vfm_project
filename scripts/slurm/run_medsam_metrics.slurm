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

# Load MedSAM
print('Loading MedSAM...')
from segment_anything import sam_model_registry

checkpoint_path = '$PROJECT_ROOT/models/medsam_checkpoints/medsam_vit_b.pth'
if not os.path.exists(checkpoint_path):
    print(f'ERROR: MedSAM checkpoint not found at {checkpoint_path}')
    sys.exit(1)

sam_model = sam_model_registry['vit_b'](checkpoint=checkpoint_path)
sam_model = sam_model.to(device)
sam_model.eval()
print('MedSAM loaded!')

def preprocess_for_medsam(image):
    \"\"\"Preprocess image for MedSAM (resize to 1024x1024).\"\"\"
    from torchvision.transforms import Resize
    import torch.nn.functional as F
    
    h, w = image.shape[:2]
    target_size = 1024
    
    img_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
    img_tensor = F.interpolate(img_tensor.unsqueeze(0), size=(target_size, target_size), mode='bilinear', align_corners=False)
    img_tensor = img_tensor.squeeze(0)
    
    # Normalize
    pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
    pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)
    img_tensor = (img_tensor - pixel_mean) / pixel_std
    
    return img_tensor, h, w

def run_medsam(image, box, sam_model, device):
    \"\"\"Run MedSAM inference with box prompt.\"\"\"
    img_tensor, orig_h, orig_w = preprocess_for_medsam(image)
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    # Scale box to 1024x1024
    scale_x = 1024 / orig_w
    scale_y = 1024 / orig_h
    scaled_box = np.array([
        box[0] * scale_x, box[1] * scale_y,
        box[2] * scale_x, box[3] * scale_y
    ])
    
    box_torch = torch.from_numpy(scaled_box).float().unsqueeze(0).to(device)
    
    with torch.no_grad():
        image_embedding = sam_model.image_encoder(img_tensor)
        
        sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
            points=None,
            boxes=box_torch,
            masks=None
        )
        
        low_res_masks, _ = sam_model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False
        )
    
    # Resize back to original size
    import torch.nn.functional as F
    mask = F.interpolate(low_res_masks, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
    mask = (mask.squeeze().cpu().numpy() > 0).astype(np.uint8)
    
    return mask

def run_medsam_with_tta(image, box, sam_model, device):
    \"\"\"Run MedSAM with test-time augmentation (flips).\"\"\"
    masks = []
    
    # Original
    masks.append(run_medsam(image, box, sam_model, device))
    
    # Horizontal flip
    img_flip_h = np.fliplr(image).copy()
    h, w = image.shape[:2]
    box_flip_h = [w - box[2], box[1], w - box[0], box[3]]
    mask_h = run_medsam(img_flip_h, box_flip_h, sam_model, device)
    masks.append(np.fliplr(mask_h))
    
    # Vertical flip
    img_flip_v = np.flipud(image).copy()
    box_flip_v = [box[0], h - box[3], box[2], h - box[1]]
    mask_v = run_medsam(img_flip_v, box_flip_v, sam_model, device)
    masks.append(np.flipud(mask_v))
    
    # Combine with majority voting
    combined = np.mean(masks, axis=0)
    return (combined > 0.5).astype(np.uint8)

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
                    pred_mask = run_medsam_with_tta(image, box, sam_model, device)
                else:
                    pred_mask = run_medsam(image, box, sam_model, device)
                
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
