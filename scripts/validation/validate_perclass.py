#!/usr/bin/env python3
"""
Per-class validation for SAM2 training.
Evaluates each tissue class separately (realistic metric).
"""

import os
import sys
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import json

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)
sam2_path = os.path.join(project_root, 'sam2')
sys.path.insert(0, sam2_path)

from src.dataset import BCSSDataset
from src.sam_segmentation import get_sam2_predictor, get_predicted_mask_from_prompts
from sam2.build_sam import build_sam2


def calculate_dice(pred_mask, gt_mask):
    """Calculate Dice coefficient for binary masks."""
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)
    
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    pred_sum = pred_mask.sum()
    gt_sum = gt_mask.sum()
    
    if pred_sum + gt_sum == 0:
        return 1.0 if intersection == 0 else 0.0
    
    dice = (2.0 * intersection) / (pred_sum + gt_sum)
    return dice


def validate_perclass(checkpoint_path, model_cfg, data_root, split='val', num_samples=None):
    """
    Run per-class validation.
    
    Args:
        checkpoint_path: Path to SAM2 checkpoint
        model_cfg: Path to SAM2 config file
        data_root: Root directory of BCSS dataset
        split: 'val' or 'test'
        num_samples: Limit number of samples (for quick testing)
    
    Returns:
        Dictionary of per-class Dice scores
    """
    print(f"\n{'='*60}")
    print(f"Per-Class Validation: {checkpoint_path}")
    print(f"{'='*60}\n")
    
    # Load SAM2 model
    print("Loading SAM2 model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = get_sam2_predictor(model_cfg, checkpoint_path, device=device)
    
    # Load dataset
    print(f"Loading {split} dataset...")
    image_dir = os.path.join(data_root, 'images')
    mask_dir = os.path.join(data_root, 'masks')
    dataset = BCSSDataset(image_dir=image_dir, mask_dir=mask_dir, split=split)
    
    if num_samples:
        dataset.image_files = dataset.image_files[:num_samples]
    
    print(f"Evaluating {len(dataset)} samples\n")
    
    # Class mapping (based on BCSS dataset)
    class_names = dataset.class_names
    
    # Store per-class scores (dynamically from dataset)
    class_dice_scores = {}
    
    # Iterate through dataset
    for idx in tqdm(range(len(dataset)), desc="Validating"):
        sample = dataset[idx]
        image_np = sample['image_np']  # [H, W, 3]
        gt_mask = sample['mask']       # [H, W] with class IDs
        
        # Convert to numpy if tensor
        if torch.is_tensor(gt_mask):
            gt_mask = gt_mask.numpy()
        
        unique_classes = sample['unique_classes']
        if torch.is_tensor(unique_classes):
            unique_classes = unique_classes.numpy()
        
        # Set image in predictor
        predictor.set_image(image_np)
        
        # Evaluate each class present in the image
        for class_id in unique_classes:
            if class_id == 0:  # Skip background
                continue
            
            if class_id not in class_names:
                continue  # Skip unknown classes
            
            # Initialize list for this class if not seen before
            if class_id not in class_dice_scores:
                class_dice_scores[class_id] = []
            
            # Create binary ground truth for this class
            binary_gt_mask = (gt_mask == class_id).astype(np.uint8)
            
            # Generate box prompt from ground truth
            coords = np.argwhere(binary_gt_mask > 0)
            if len(coords) == 0:
                continue
            
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            box = np.array([x_min, y_min, x_max, y_max])
            
            # Get SAM prediction
            masks, _, _ = predictor.predict(
                box=box,
                multimask_output=False
            )
            pred_mask = masks[0].astype(np.uint8)
            
            # Calculate Dice
            dice = calculate_dice(pred_mask, binary_gt_mask)
            class_dice_scores[class_id].append(dice)
    
    # Compute average per-class Dice
    results = {}
    print(f"\n{'='*60}")
    print("Per-Class Results:")
    print(f"{'='*60}")
    
    for class_id, class_name in class_names.items():
        if class_id == 0:  # Skip background in reporting
            continue
        
        if class_id in class_dice_scores and len(class_dice_scores[class_id]) > 0:
            scores = class_dice_scores[class_id]
            avg_dice = np.mean(scores)
            std_dice = np.std(scores)
            results[class_name] = {
                'dice': float(avg_dice),
                'std': float(std_dice),
                'count': len(scores)
            }
            print(f"{class_name:15s}: Dice {avg_dice:.4f} ± {std_dice:.4f} ({len(scores)} samples)")
        else:
            results[class_name] = {'dice': 0.0, 'std': 0.0, 'count': 0}
            print(f"{class_name:15s}: No samples")
    
    # Compute overall average (micro-average across all classes)
    all_scores = [score for scores in class_dice_scores.values() for score in scores]
    overall_dice = np.mean(all_scores) if all_scores else 0.0
    
    print(f"{'='*60}")
    print(f"Overall Per-Class Dice: {overall_dice:.4f}")
    print(f"{'='*60}\n")
    
    results['overall'] = float(overall_dice)
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Per-class validation for SAM2')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to SAM2 checkpoint')
    parser.add_argument('--model_cfg', type=str, 
                        default='configs/sam2.1/sam2.1_hiera_l.yaml',
                        help='Path to SAM2 model config (relative to sam2 package)')
    parser.add_argument('--data_root', type=str, default='data/bcss',
                        help='Root directory of BCSS dataset')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'],
                        help='Dataset split to evaluate')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Limit number of samples (for quick testing)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for results')
    
    args = parser.parse_args()
    
    # Resolve relative paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    
    # Checkpoint needs absolute path
    if not os.path.isabs(args.checkpoint):
        args.checkpoint = os.path.join(project_root, args.checkpoint)
    
    # Model config should be relative to sam2 package (like "configs/sam2.1/sam2.1_hiera_l.yaml")
    # Don't convert to absolute path - Hydra needs relative path
    
    args.data_root = os.path.join(project_root, args.data_root)
    
    # Run validation
    results = validate_perclass(
        checkpoint_path=args.checkpoint,
        model_cfg=args.model_cfg,
        data_root=args.data_root,
        split=args.split,
        num_samples=args.num_samples
    )
    
    # Save results if requested
    if args.output:
        output_path = os.path.join(project_root, args.output)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {output_path}")
