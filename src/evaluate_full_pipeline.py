#!/usr/bin/env python3
"""
Full Pipeline Evaluation: SAM2 Segmentation + CLIP Classification
Evaluates all configurations with:
- Segmentation metrics: Dice and IoU
- Classification metrics: Accuracy, Per-class accuracy
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
from clip_classification import CLIPClassifier, crop_region_from_mask, load_prompts_from_json
from device_utils import get_device
from training.utils.train_utils import register_omegaconf_resolvers


def evaluate_configuration(
    predictor,
    clip_classifier,
    clip_prompts,
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
    Evaluate a specific configuration with both segmentation and classification metrics.
    
    Returns dict with:
    - Per-class Dice, IoU
    - Overall Dice, IoU
    - Classification accuracy (overall and per-class)
    """
    from src.sam_segmentation import get_predicted_mask_from_prompts
    from src.tta_utils import predict_with_tta
    from src.iterative_refinement import predict_with_refinement, predict_with_tta_and_refinement
    
    class_names = bcss_dataset.class_names
    
    # Segmentation metrics storage
    class_dice_scores = {cid: [] for cid in class_names if cid != 0}
    class_iou_scores = {cid: [] for cid in class_names if cid != 0}
    
    # Classification metrics storage
    classification_correct = {cid: 0 for cid in class_names if cid != 0}
    classification_total = {cid: 0 for cid in class_names if cid != 0}
    
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
            
            # Calculate segmentation metrics
            dice, iou = calculate_metrics(predicted_mask, binary_gt_mask)
            class_dice_scores[class_id].append(dice)
            class_iou_scores[class_id].append(iou)
            
            # CLIP Classification on predicted mask
            if clip_classifier is not None and predicted_mask.sum() > 0:
                cropped_region = crop_region_from_mask(image, predicted_mask)
                if cropped_region is not None:
                    predicted_class = clip_classifier.classify_region(cropped_region, clip_prompts)
                    classification_total[class_id] += 1
                    if predicted_class == class_name:
                        classification_correct[class_id] += 1
    
    # Aggregate results
    results = {
        'segmentation': {},
        'classification': {},
        'per_class': {}
    }
    
    # Per-class results
    for cid, cname in class_names.items():
        if cid == 0:
            continue
        
        dice_scores = class_dice_scores.get(cid, [])
        iou_scores = class_iou_scores.get(cid, [])
        
        correct = classification_correct.get(cid, 0)
        total = classification_total.get(cid, 0)
        
        results['per_class'][cname] = {
            'dice': float(np.mean(dice_scores)) if dice_scores else 0,
            'dice_std': float(np.std(dice_scores)) if dice_scores else 0,
            'iou': float(np.mean(iou_scores)) if iou_scores else 0,
            'iou_std': float(np.std(iou_scores)) if iou_scores else 0,
            'classification_accuracy': float(correct / total) if total > 0 else 0,
            'classification_correct': correct,
            'classification_total': total,
            'seg_count': len(dice_scores)
        }
    
    # Overall segmentation metrics
    all_dice = [s for scores in class_dice_scores.values() for s in scores]
    all_iou = [s for scores in class_iou_scores.values() for s in scores]
    
    results['segmentation']['overall_dice'] = float(np.mean(all_dice)) if all_dice else 0
    results['segmentation']['overall_iou'] = float(np.mean(all_iou)) if all_iou else 0
    results['segmentation']['dice_std'] = float(np.std(all_dice)) if all_dice else 0
    results['segmentation']['iou_std'] = float(np.std(all_iou)) if all_iou else 0
    results['segmentation']['total_samples'] = len(all_dice)
    
    # Overall classification metrics
    total_correct = sum(classification_correct.values())
    total_samples = sum(classification_total.values())
    results['classification']['overall_accuracy'] = float(total_correct / total_samples) if total_samples > 0 else 0
    results['classification']['total_correct'] = total_correct
    results['classification']['total_samples'] = total_samples
    
    return results


def load_sam2_predictor(model_cfg, checkpoint, device):
    """Load SAM2 predictor with proper Hydra config handling."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sam2_root = os.path.join(project_root, 'sam2')
    checkpoint_path = os.path.join(project_root, checkpoint) if not os.path.isabs(checkpoint) else checkpoint
    
    orig_cwd = os.getcwd()
    os.chdir(sam2_root)
    
    try:
        predictor = get_sam2_predictor(model_cfg, checkpoint_path, device)
    finally:
        os.chdir(orig_cwd)
    
    return predictor


def main():
    parser = argparse.ArgumentParser(description="Full pipeline evaluation: SAM2 + CLIP")
    parser.add_argument('--model_cfg', type=str, 
                        default="configs/sam2.1/sam2.1_hiera_l.yaml",
                        help='SAM2 model config')
    parser.add_argument('--checkpoint', type=str,
                        default="sam2/checkpoints/sam2.1_hiera_large.pt",
                        help='SAM2 checkpoint path')
    parser.add_argument('--clip_prompts', type=str,
                        default="configs/prompts/hard_coded_prompts_v2.json",
                        help='CLIP prompts JSON file')
    parser.add_argument('--split', type=str, default='test', help='Dataset split')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--output_dir', type=str, default='results/full_pipeline_eval')
    parser.add_argument('--quick', action='store_true', help='Quick test with 5 samples')
    parser.add_argument('--no_clip', action='store_true', help='Skip CLIP classification')
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
    
    # Load SAM2 predictor
    predictor = load_sam2_predictor(args.model_cfg, args.checkpoint, device)
    print("SAM2 predictor loaded")
    
    # Load CLIP classifier
    clip_classifier = None
    clip_prompts = None
    if not args.no_clip:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        clip_prompts_path = os.path.join(project_root, args.clip_prompts)
        clip_prompts = load_prompts_from_json(clip_prompts_path)
        clip_classifier = CLIPClassifier(device=device)
        print("CLIP classifier loaded")
    
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
    
    print("\n" + "="*80)
    print("Full Pipeline Evaluation: SAM2 Segmentation + CLIP Classification")
    print("="*80)
    
    for cfg in configs:
        print(f"\n>>> Testing: {cfg['name']}")
        start_time = time.time()
        
        results = evaluate_configuration(
            predictor=predictor,
            clip_classifier=clip_classifier,
            clip_prompts=clip_prompts,
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
        
        seg = results['segmentation']
        clf = results['classification']
        print(f"    Dice: {seg['overall_dice']:.4f} | IoU: {seg['overall_iou']:.4f} | "
              f"CLIP Acc: {clf['overall_accuracy']:.2%} | Time: {elapsed:.1f}s")
    
    # ========== SUMMARY ==========
    print("\n" + "="*80)
    print("SEGMENTATION RESULTS SUMMARY")
    print("="*80)
    print(f"{'Configuration':<25} {'Dice':>10} {'IoU':>10} {'vs Baseline':>12}")
    print("-"*60)
    
    baseline_dice = all_results['box_neg_baseline']['segmentation']['overall_dice']
    for name, results in all_results.items():
        dice = results['segmentation']['overall_dice']
        iou = results['segmentation']['overall_iou']
        diff = (dice - baseline_dice) / baseline_dice * 100 if baseline_dice > 0 else 0
        diff_str = f"+{diff:.1f}%" if diff >= 0 else f"{diff:.1f}%"
        print(f"{name:<25} {dice:>10.4f} {iou:>10.4f} {diff_str:>12}")
    
    if not args.no_clip:
        print("\n" + "="*80)
        print("CLASSIFICATION RESULTS SUMMARY")
        print("="*80)
        print(f"{'Configuration':<25} {'Accuracy':>12} {'Correct/Total':>15}")
        print("-"*55)
        
        for name, results in all_results.items():
            clf = results['classification']
            acc = clf['overall_accuracy']
            correct = clf['total_correct']
            total = clf['total_samples']
            print(f"{name:<25} {acc:>11.2%} {correct:>7}/{total:<7}")
    
    # Per-class breakdown for best config
    print("\n" + "="*80)
    print("PER-CLASS BREAKDOWN (Best Configuration)")
    print("="*80)
    
    best_config = max(all_results.keys(), 
                      key=lambda k: all_results[k]['segmentation']['overall_dice'])
    best = all_results[best_config]
    baseline = all_results['box_neg_baseline']
    
    print(f"\nBest config: {best_config}")
    print(f"{'Class':<15} {'Dice':>8} {'IoU':>8} {'CLIP Acc':>10} {'Δ Dice':>10}")
    print("-"*55)
    
    for class_name in ['tumor', 'stroma', 'lymphocyte', 'necrosis', 'blood_vessel']:
        if class_name in best['per_class']:
            pc = best['per_class'][class_name]
            base_pc = baseline['per_class'].get(class_name, {})
            
            dice = pc['dice']
            iou = pc['iou']
            acc = pc['classification_accuracy']
            base_dice = base_pc.get('dice', 0)
            diff = dice - base_dice
            diff_str = f"+{diff:.4f}" if diff >= 0 else f"{diff:.4f}"
            
            print(f"{class_name:<15} {dice:>8.4f} {iou:>8.4f} {acc:>9.2%} {diff_str:>10}")
    
    # Save results
    output_path = os.path.join(args.output_dir, 'full_pipeline_results.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    
    # Also save a simple summary CSV
    csv_path = os.path.join(args.output_dir, 'summary.csv')
    with open(csv_path, 'w') as f:
        f.write("Configuration,Dice,IoU,CLIP_Accuracy,Time_s\n")
        for name, results in all_results.items():
            seg = results['segmentation']
            clf = results['classification']
            f.write(f"{name},{seg['overall_dice']:.4f},{seg['overall_iou']:.4f},"
                    f"{clf['overall_accuracy']:.4f},{results['elapsed_seconds']:.1f}\n")
    print(f"Summary CSV saved to: {csv_path}")


if __name__ == '__main__':
    main()
