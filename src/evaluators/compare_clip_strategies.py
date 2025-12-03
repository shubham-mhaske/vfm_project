#!/usr/bin/env python3
"""
CLIP Classification Strategy Comparison
=========================================
Compares different cropping and overlay strategies to find optimal 
SAM + CLIP integration.

Key experiments:
1. Crop strategies: tight vs context vs square
2. Overlay strategies: none vs blur vs dim vs masked
3. Multi-crop ensemble
4. Temperature scaling for confidence calibration
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'sam2'))

import argparse
import json
import numpy as np
from tqdm import tqdm
import torch
from datetime import datetime
from collections import defaultdict

from dataset import BCSSDataset
from sam_segmentation import get_prompts_from_mask, calculate_metrics
from clip_classification_v2 import (
    CLIPClassifierV2, CropStrategy, MaskOverlay,
    load_prompts_from_json, crop_region_from_mask
)
from device_utils import get_device
from training.utils.train_utils import register_omegaconf_resolvers


def log(msg):
    """Print with timestamp."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def evaluate_clip_strategy(
    bcss_dataset,
    clip_classifier: CLIPClassifierV2,
    clip_prompts: dict,
    crop_strategy: str = 'tight',
    overlay_strategy: str = 'none',
    use_ensemble: bool = False,
    max_samples: int = None,
    use_gt_masks: bool = True
):
    """
    Evaluate a specific CLIP classification strategy.
    Uses GT masks for isolation (to test CLIP in ideal conditions).
    """
    class_names = bcss_dataset.class_names
    
    # Track per-class performance
    correct = defaultdict(int)
    total = defaultdict(int)
    confidences = defaultdict(list)
    confusion = defaultdict(lambda: defaultdict(int))  # confusion[true][pred]
    
    num_samples = len(bcss_dataset) if max_samples is None else min(max_samples, len(bcss_dataset))
    
    for i in tqdm(range(num_samples), desc=f"{crop_strategy}+{overlay_strategy}"):
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
            
            # Use ground truth mask for this experiment
            binary_mask = (gt_mask == class_id).astype(np.uint8)
            
            if binary_mask.sum() < 100:  # Skip very small regions
                continue
            
            # Classify using the specified strategy
            if use_ensemble:
                pred_class, conf, probs = clip_classifier.classify_with_ensemble(
                    image, binary_mask, clip_prompts,
                    crop_strategies=[crop_strategy, 'context'],
                    overlay_strategy=overlay_strategy
                )
            else:
                # Single crop approach
                cropped = crop_region_from_mask(
                    image, binary_mask, 
                    strategy=crop_strategy, 
                    overlay=overlay_strategy
                )
                if cropped is None:
                    continue
                pred_class, conf, probs = clip_classifier.classify_region(cropped, clip_prompts)
            
            if pred_class is None:
                continue
            
            # Track metrics
            total[class_name] += 1
            confidences[class_name].append(conf)
            confusion[class_name][pred_class] += 1
            
            if pred_class == class_name:
                correct[class_name] += 1
    
    # Calculate aggregate metrics
    results = {
        'strategy': f"{crop_strategy}+{overlay_strategy}",
        'use_ensemble': use_ensemble,
        'per_class': {},
        'overall': {}
    }
    
    total_correct = 0
    total_samples = 0
    all_confidences = []
    
    for cn in ['tumor', 'stroma', 'lymphocyte', 'necrosis', 'blood_vessel']:
        if cn in total and total[cn] > 0:
            acc = correct[cn] / total[cn]
            avg_conf = np.mean(confidences[cn]) if confidences[cn] else 0
            
            results['per_class'][cn] = {
                'accuracy': float(acc),
                'correct': correct[cn],
                'total': total[cn],
                'avg_confidence': float(avg_conf),
                'confusion': dict(confusion[cn])
            }
            
            total_correct += correct[cn]
            total_samples += total[cn]
            all_confidences.extend(confidences[cn])
    
    results['overall']['accuracy'] = float(total_correct / total_samples) if total_samples > 0 else 0
    results['overall']['total_correct'] = total_correct
    results['overall']['total_samples'] = total_samples
    results['overall']['avg_confidence'] = float(np.mean(all_confidences)) if all_confidences else 0
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Compare CLIP classification strategies")
    parser.add_argument('--clip_prompts', type=str,
                        default="configs/prompts/hard_coded_prompts_v2.json")
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--output_dir', type=str, default='results/clip_strategy_comparison')
    parser.add_argument('--quick', action='store_true', help='Quick test with 10 samples')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Softmax temperature for calibration')
    args = parser.parse_args()
    
    if args.quick:
        args.max_samples = 10
    
    log("Starting CLIP Strategy Comparison")
    log("="*60)
    
    device = get_device()
    log(f"Device: {device}")
    
    # Load dataset
    log("Loading BCSS dataset...")
    bcss_dataset = BCSSDataset(
        image_dir='data/bcss/images',
        mask_dir='data/bcss/masks',
        split=args.split
    )
    log(f"Loaded {len(bcss_dataset)} samples")
    
    # Load CLIP
    log("Loading CLIPClassifierV2...")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    clip_prompts_path = os.path.join(project_root, args.clip_prompts)
    clip_prompts = load_prompts_from_json(clip_prompts_path)
    
    clip_classifier = CLIPClassifierV2(
        device=device,
        temperature=args.temperature
    )
    log("CLIP loaded!")
    
    # Output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Strategies to test
    crop_strategies = ['tight', 'context', 'square']
    overlay_strategies = ['none', 'blur', 'dim', 'masked']
    
    all_results = {}
    
    # ============ EXPERIMENT 1: Crop Strategies (no overlay) ============
    log("\n" + "="*60)
    log("EXPERIMENT 1: Comparing Crop Strategies (no overlay)")
    log("="*60)
    
    for crop_strat in crop_strategies:
        log(f"\nTesting: {crop_strat} crop")
        results = evaluate_clip_strategy(
            bcss_dataset, clip_classifier, clip_prompts,
            crop_strategy=crop_strat,
            overlay_strategy='none',
            max_samples=args.max_samples
        )
        key = f"crop_{crop_strat}"
        all_results[key] = results
        log(f"  Accuracy: {results['overall']['accuracy']:.2%}")
    
    # ============ EXPERIMENT 2: Overlay Strategies (tight crop) ============
    log("\n" + "="*60)
    log("EXPERIMENT 2: Comparing Overlay Strategies (tight crop)")
    log("="*60)
    
    for overlay_strat in overlay_strategies:
        log(f"\nTesting: {overlay_strat} overlay")
        results = evaluate_clip_strategy(
            bcss_dataset, clip_classifier, clip_prompts,
            crop_strategy='tight',
            overlay_strategy=overlay_strat,
            max_samples=args.max_samples
        )
        key = f"overlay_{overlay_strat}"
        all_results[key] = results
        log(f"  Accuracy: {results['overall']['accuracy']:.2%}")
    
    # ============ EXPERIMENT 3: Best Combinations ============
    log("\n" + "="*60)
    log("EXPERIMENT 3: Testing Best Combinations")
    log("="*60)
    
    # Based on histopathology intuition: context + blur might work well
    combos = [
        ('context', 'blur'),
        ('context', 'dim'),
        ('tight', 'blur'),
        ('square', 'none'),
    ]
    
    for crop_strat, overlay_strat in combos:
        log(f"\nTesting: {crop_strat} + {overlay_strat}")
        results = evaluate_clip_strategy(
            bcss_dataset, clip_classifier, clip_prompts,
            crop_strategy=crop_strat,
            overlay_strategy=overlay_strat,
            max_samples=args.max_samples
        )
        key = f"combo_{crop_strat}_{overlay_strat}"
        all_results[key] = results
        log(f"  Accuracy: {results['overall']['accuracy']:.2%}")
    
    # ============ EXPERIMENT 4: Ensemble ============
    log("\n" + "="*60)
    log("EXPERIMENT 4: Multi-Crop Ensemble")
    log("="*60)
    
    log("\nTesting: Ensemble (tight + context)")
    results = evaluate_clip_strategy(
        bcss_dataset, clip_classifier, clip_prompts,
        crop_strategy='tight',
        overlay_strategy='none',
        use_ensemble=True,
        max_samples=args.max_samples
    )
    all_results['ensemble_tight_context'] = results
    log(f"  Accuracy: {results['overall']['accuracy']:.2%}")
    
    # ============ SUMMARY ============
    log("\n" + "="*70)
    log("RESULTS SUMMARY")
    log("="*70)
    
    # Sort by accuracy
    sorted_results = sorted(
        all_results.items(),
        key=lambda x: x[1]['overall']['accuracy'],
        reverse=True
    )
    
    log(f"\n{'Strategy':<35} {'Accuracy':>10} {'Confidence':>12}")
    log("-"*60)
    
    for name, res in sorted_results:
        acc = res['overall']['accuracy']
        conf = res['overall']['avg_confidence']
        log(f"{name:<35} {acc:>9.2%} {conf:>11.3f}")
    
    # Per-class breakdown for top 3
    log("\n" + "="*70)
    log("PER-CLASS BREAKDOWN (Top 3 Strategies)")
    log("="*70)
    
    for name, res in sorted_results[:3]:
        log(f"\n{name}:")
        log(f"  {'Class':<15} {'Accuracy':>10} {'Correct':>10} {'Total':>8}")
        log("  " + "-"*45)
        for cn in ['tumor', 'stroma', 'lymphocyte', 'necrosis', 'blood_vessel']:
            if cn in res['per_class']:
                pc = res['per_class'][cn]
                log(f"  {cn:<15} {pc['accuracy']:>9.2%} {pc['correct']:>10} {pc['total']:>8}")
    
    # Save results
    output_path = os.path.join(args.output_dir, 'strategy_comparison.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    log(f"\nResults saved to: {output_path}")
    
    # Best recommendation
    best_name, best_res = sorted_results[0]
    log("\n" + "="*70)
    log(f"RECOMMENDATION: Use '{best_name}' strategy")
    log(f"  Overall Accuracy: {best_res['overall']['accuracy']:.2%}")
    log("="*70)


if __name__ == '__main__':
    main()
