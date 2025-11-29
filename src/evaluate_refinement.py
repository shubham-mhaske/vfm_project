#!/usr/bin/env python3
"""
Evaluate iterative mask refinement on BCSS dataset.
Compares: baseline, refinement only, TTA only, TTA + refinement
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'sam2'))

import argparse
import json
import time
import numpy as np
from tqdm import tqdm
import torch

from dataset import BCSSDataset
from sam_segmentation import get_sam2_predictor, get_prompts_from_mask, calculate_metrics
from device_utils import get_device
from training.utils.train_utils import register_omegaconf_resolvers


def evaluate_configuration(
    predictor,
    bcss_dataset,
    prompt_type: str = 'box',
    use_neg_points: bool = True,
    use_tta: bool = False,
    use_refinement: bool = False,
    num_refinement_iterations: int = 2,
    max_samples: int = None,
    show_progress: bool = True
):
    """
    Evaluate a specific configuration and return results.
    """
    from src.sam_segmentation import get_predicted_mask_from_prompts
    from src.tta_utils import predict_with_tta
    from src.iterative_refinement import predict_with_refinement, predict_with_tta_and_refinement
    
    class_names = bcss_dataset.class_names
    class_dice_scores = {cid: [] for cid in class_names if cid != 0}
    
    num_samples = len(bcss_dataset) if max_samples is None else min(max_samples, len(bcss_dataset))
    
    for i in tqdm(range(num_samples), disable=not show_progress, desc="Evaluating"):
        sample = bcss_dataset[i]
        image = sample['image_np']
        gt_mask = sample['mask'].numpy()
        unique_classes = np.unique(gt_mask)
        
        for class_id in unique_classes:
            if class_id == 0:
                continue
            
            class_name = class_names.get(class_id)
            if not class_name:
                continue
            
            binary_gt_mask = (gt_mask == class_id).astype(np.uint8)
            prompts = get_prompts_from_mask(binary_gt_mask)
            
            if prompt_type not in prompts:
                continue
            
            # Select prediction method based on configuration
            if use_tta and use_refinement:
                predicted_mask = predict_with_tta_and_refinement(
                    predictor, image, prompts, prompt_type, use_neg_points,
                    num_augmentations=4, num_refinement_iterations=num_refinement_iterations
                )
            elif use_tta:
                predicted_mask = predict_with_tta(
                    predictor, image, prompts, prompt_type, use_neg_points
                )
            elif use_refinement:
                predicted_mask = predict_with_refinement(
                    predictor, image, prompts, prompt_type, use_neg_points,
                    num_iterations=num_refinement_iterations
                )
            else:
                # Baseline: single prediction
                predictor.set_image(image)
                mask_logits, _, _ = get_predicted_mask_from_prompts(
                    predictor, image, prompts, prompt_type, use_neg_points
                )
                predicted_mask = (mask_logits > 0.5).astype(np.uint8)
            
            dice, iou = calculate_metrics(predicted_mask, binary_gt_mask)
            class_dice_scores[class_id].append(dice)
    
    # Aggregate results
    results = {}
    for cid, cname in class_names.items():
        if cid == 0:
            continue
        scores = class_dice_scores.get(cid, [])
        results[cname] = {
            'dice': float(np.mean(scores)) if scores else 0,
            'std': float(np.std(scores)) if scores else 0,
            'count': len(scores)
        }
    
    all_scores = [s for scores in class_dice_scores.values() for s in scores]
    results['overall'] = float(np.mean(all_scores)) if all_scores else 0
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate iterative refinement on BCSS")
    parser.add_argument('--model_cfg', type=str, 
                        default="configs/sam2.1/sam2.1_hiera_l.yaml",
                        help='SAM2 model config (relative to sam2 package)')
    parser.add_argument('--checkpoint', type=str,
                        default="sam2/checkpoints/sam2.1_hiera_large.pt",
                        help='SAM2 checkpoint path (relative to project root)')
    parser.add_argument('--split', type=str, default='test', help='Dataset split')
    parser.add_argument('--max_samples', type=int, default=None, help='Max samples to evaluate')
    parser.add_argument('--output_dir', type=str, default='results/refinement_experiments',
                        help='Output directory for results')
    parser.add_argument('--quick', action='store_true', help='Quick test with 5 samples')
    args = parser.parse_args()
    
    if args.quick:
        args.max_samples = 5
    
    # Setup
    device = get_device()
    print(f"Using device: {device}")
    register_omegaconf_resolvers()
    
    # Load dataset
    bcss_dataset = BCSSDataset(
        image_dir='data/bcss/images',
        mask_dir='data/bcss/masks',
        split=args.split
    )
    print(f"Loaded {len(bcss_dataset)} samples from {args.split} split")
    
    # Load predictor
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sam2_root = os.path.join(project_root, 'sam2')
    checkpoint = os.path.join(project_root, args.checkpoint)
    
    # Change to sam2 directory for Hydra config resolution
    orig_cwd = os.getcwd()
    os.chdir(sam2_root)
    
    try:
        # model_cfg should be relative to sam2/sam2/configs (the config search path)
        # Use format: "configs/sam2.1/sam2.1_hiera_l.yaml" for sam2.1 models
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        predictor = get_sam2_predictor(model_cfg, checkpoint, device)
        print("SAM2 predictor loaded")
    finally:
        os.chdir(orig_cwd)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configurations to test
    configs = [
        {'name': 'box_neg_baseline', 'tta': False, 'refine': False, 'iters': 1},
        {'name': 'box_neg_refine2', 'tta': False, 'refine': True, 'iters': 2},
        {'name': 'box_neg_refine3', 'tta': False, 'refine': True, 'iters': 3},
        {'name': 'box_neg_tta', 'tta': True, 'refine': False, 'iters': 1},
        {'name': 'box_neg_tta_refine2', 'tta': True, 'refine': True, 'iters': 2},
    ]
    
    all_results = {}
    
    print("\n" + "="*60)
    print("Iterative Refinement Experiment")
    print("="*60)
    
    for cfg in configs:
        print(f"\n>>> Testing: {cfg['name']}")
        start_time = time.time()
        
        results = evaluate_configuration(
            predictor=predictor,
            bcss_dataset=bcss_dataset,
            prompt_type='box',
            use_neg_points=True,
            use_tta=cfg['tta'],
            use_refinement=cfg['refine'],
            num_refinement_iterations=cfg['iters'],
            max_samples=args.max_samples,
            show_progress=True
        )
        
        elapsed = time.time() - start_time
        results['elapsed_seconds'] = elapsed
        all_results[cfg['name']] = results
        
        print(f"    Overall Dice: {results['overall']:.4f}")
        print(f"    Time: {elapsed:.1f}s")
    
    # Summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"{'Configuration':<25} {'Overall Dice':>12} {'vs Baseline':>12}")
    print("-"*60)
    
    baseline_dice = all_results['box_neg_baseline']['overall']
    for name, results in all_results.items():
        dice = results['overall']
        diff = (dice - baseline_dice) / baseline_dice * 100 if baseline_dice > 0 else 0
        diff_str = f"+{diff:.1f}%" if diff >= 0 else f"{diff:.1f}%"
        print(f"{name:<25} {dice:>12.4f} {diff_str:>12}")
    
    # Save results
    output_path = os.path.join(args.output_dir, 'refinement_results.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    
    # Per-class breakdown for best configs
    print("\n" + "="*60)
    print("PER-CLASS BREAKDOWN (Best vs Baseline)")
    print("="*60)
    
    best_config = max(all_results.keys(), key=lambda k: all_results[k]['overall'])
    baseline = all_results['box_neg_baseline']
    best = all_results[best_config]
    
    print(f"\nBest config: {best_config}")
    print(f"{'Class':<15} {'Baseline':>10} {'Best':>10} {'Diff':>10}")
    print("-"*45)
    
    for class_name in ['tumor', 'stroma', 'lymphocyte', 'necrosis', 'blood_vessel']:
        if class_name in baseline:
            base_dice = baseline[class_name]['dice']
            best_dice = best[class_name]['dice']
            diff = best_dice - base_dice
            diff_str = f"+{diff:.4f}" if diff >= 0 else f"{diff:.4f}"
            print(f"{class_name:<15} {base_dice:>10.4f} {best_dice:>10.4f} {diff_str:>10}")


if __name__ == '__main__':
    main()
