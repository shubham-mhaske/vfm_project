#!/bin/bash
#SBATCH --job-name=clip_metrics
#SBATCH --output=slurm_logs/clip_metrics_%j.out
#SBATCH --error=slurm_logs/clip_metrics_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:30:00

# CLIP Classification Metrics Collection
# Evaluates 8 prompt configurations on test set

echo "=========================================="
echo "CLIP CLASSIFICATION METRICS"
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
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

# Create output directory
mkdir -p results/complete_metrics
mkdir -p slurm_logs

echo "Running CLIP evaluation..."
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
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from dataset import BCSSDataset
from clip_classification import CLIPClassifier, crop_region_from_mask, load_prompts_from_json
from sam_segmentation import get_prompts_from_mask

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')
if device == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')

# Class mappings
class_names = {0: 'background', 1: 'tumor', 2: 'stroma', 3: 'lymphocyte', 4: 'necrosis', 18: 'blood_vessel'}
target_class_ids = [1, 2, 3, 4, 18]
class_name_list = ['tumor', 'stroma', 'lymphocyte', 'necrosis', 'blood_vessel']

# Load dataset
print('Loading dataset...')
dataset = BCSSDataset(
    image_dir='$PROJECT_ROOT/data/bcss/images',
    mask_dir='$PROJECT_ROOT/data/bcss/masks',
    split='test'
)
print(f'Loaded {len(dataset)} test samples')

# CLIP prompt configurations
prompt_configs = [
    {'name': 'hardcoded_v1', 'path': 'configs/prompts/hard_coded_prompts.json'},
    {'name': 'hardcoded_v2', 'path': 'configs/prompts/hard_coded_prompts_v2.json'},
    {'name': 'llm_text_v1_jargon', 'path': 'configs/prompts/llm_text_prompts_v1_gemini_pro_latest.json'},
    {'name': 'llm_text_v2_clip_friendly', 'path': 'configs/prompts/llm_text_prompts_v2_clip_friendly.json'},
    {'name': 'llm_text_v3_fewshot', 'path': 'configs/prompts/llm_text_prompts_v3_fewshot.json'},
    {'name': 'llm_multimodal_v1', 'path': 'configs/prompts/llm_multimodal_prompts_v1_gemini_2.5_flash.json'},
    {'name': 'llm_multimodal_v2_clip_friendly', 'path': 'configs/prompts/llm_multimodal_prompts_v2_clip_friendly.json'},
    {'name': 'llm_multimodal_v3_fewshot', 'path': 'configs/prompts/llm_multimodal_prompts_v3_fewshot.json'},
]

all_results = {}
start_time = time.time()

for config in prompt_configs:
    config_name = config['name']
    prompt_path = os.path.join('$PROJECT_ROOT', config['path'])
    
    if not os.path.exists(prompt_path):
        print(f'WARNING: {prompt_path} not found, skipping...')
        continue
    
    print(f'\n--- Evaluating: {config_name} ---')
    
    # Load prompts
    prompts = load_prompts_from_json(prompt_path)
    
    # Initialize classifier once per prompt config
    if 'classifier' not in dir():
        classifier = CLIPClassifier(device=device)
    
    y_true, y_pred = [], []
    
    for i in tqdm(range(len(dataset)), desc=config_name):
        sample = dataset[i]
        image = sample['image_np']
        gt_mask = sample['mask'].numpy()
        
        for class_id in np.unique(gt_mask):
            if class_id == 0 or class_id not in class_names:
                continue
            
            binary_mask = (gt_mask == class_id).astype(np.uint8)
            if binary_mask.sum() < 100:  # Skip very small regions
                continue
            
            # Crop region
            crop = crop_region_from_mask(image, binary_mask)
            if crop is None:
                continue
            
            # Classify using classify_region(image, prompts)
            pred_class = classifier.classify_region(crop, prompts)
            
            true_label = class_names[class_id]
            y_true.append(true_label)
            y_pred.append(pred_class)
    
    # Calculate metrics
    if y_true:
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=class_name_list, average=None, zero_division=0
        )
        
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        
        accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
        
        results = {
            'per_class': {},
            'macro_avg': {
                'precision': float(macro_p),
                'recall': float(macro_r),
                'f1_score': float(macro_f1)
            },
            'overall': {
                'accuracy': float(accuracy),
                'total_samples': len(y_true)
            }
        }
        
        for idx, cname in enumerate(class_name_list):
            results['per_class'][cname] = {
                'precision': float(precision[idx]),
                'recall': float(recall[idx]),
                'f1_score': float(f1[idx]),
                'support': int(support[idx])
            }
        
        all_results[config_name] = results
        print(f'  Accuracy: {accuracy:.4f}, Macro F1: {macro_f1:.4f}')

elapsed = time.time() - start_time
print(f'\nCompleted in {elapsed/60:.1f} minutes')

# Save results
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_path = f'$PROJECT_ROOT/results/complete_metrics/clip_classification_{timestamp}.json'
with open(output_path, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f'Results saved to: {output_path}')
"

echo ""
echo "=========================================="
echo "CLIP metrics collection complete!"
echo "Finished: $(date)"
echo "=========================================="
