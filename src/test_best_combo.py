#!/usr/bin/env python3
"""
Quick experiment: Combine best CLIP prompts with best crop strategy (blur overlay)
Expected improvement: 38.9% (fewshot) + ~5% (blur) = ~43%?
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime

from dataset import BCSSDataset
from clip_classification_v2 import CLIPClassifierV2, load_prompts_from_json, crop_region_from_mask
from device_utils import get_device


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def evaluate_best_combo(dataset, clip_classifier, prompt_configs):
    """Evaluate CLIP with best settings: fewshot prompts + blur overlay."""
    
    results = {}
    
    for config_name, prompt_file in prompt_configs.items():
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        prompt_path = os.path.join(project_root, prompt_file)
        
        if not os.path.exists(prompt_path):
            log(f"SKIP: {config_name} - file not found")
            continue
        
        clip_prompts = load_prompts_from_json(prompt_path)
        
        # Remove background if present
        if 'background' in clip_prompts:
            del clip_prompts['background']
        
        class_names = dataset.class_names
        correct = defaultdict(int)
        total = defaultdict(int)
        
        log(f"\nTesting: {config_name} with BLUR overlay")
        
        for i in tqdm(range(len(dataset)), desc=config_name):
            sample = dataset[i]
            image = sample['image_np']
            gt_mask = sample['mask'].numpy()
            
            for class_id in np.unique(gt_mask):
                if class_id == 0:
                    continue
                
                class_name = class_names.get(class_id)
                if not class_name or class_name not in clip_prompts:
                    continue
                
                binary_mask = (gt_mask == class_id).astype(np.uint8)
                
                if binary_mask.sum() < 100:
                    continue
                
                # Use BLUR overlay (best from our experiments)
                cropped = crop_region_from_mask(
                    image, binary_mask,
                    strategy='tight',
                    overlay='blur'  # Key improvement!
                )
                
                if cropped is None:
                    continue
                
                pred_class, conf, _ = clip_classifier.classify_region(cropped, clip_prompts)
                
                total[class_name] += 1
                if pred_class == class_name:
                    correct[class_name] += 1
        
        # Calculate accuracy
        total_correct = sum(correct.values())
        total_samples = sum(total.values())
        overall_acc = total_correct / total_samples if total_samples > 0 else 0
        
        results[config_name] = {
            'accuracy': overall_acc,
            'correct': total_correct,
            'total': total_samples,
            'per_class': {cn: correct[cn]/total[cn] if total[cn] > 0 else 0 
                          for cn in correct.keys()}
        }
        
        log(f"  Accuracy: {overall_acc:.2%} ({total_correct}/{total_samples})")
    
    return results


def main():
    log("="*70)
    log("BEST COMBO EXPERIMENT: Best Prompts + Blur Overlay")
    log("="*70)
    
    device = get_device()
    log(f"Device: {device}")
    
    # Load dataset
    log("Loading dataset...")
    dataset = BCSSDataset('data/bcss/images', 'data/bcss/masks', split='test')
    log(f"Loaded {len(dataset)} samples")
    
    # Load CLIP
    log("Loading CLIPClassifierV2...")
    clip_classifier = CLIPClassifierV2(device=device)
    log("CLIP loaded!")
    
    # Test configurations
    prompt_configs = {
        'fewshot_blur': 'configs/prompts/llm_text_prompts_v3_fewshot.json',
        'hardcoded_v2_blur': 'configs/prompts/hard_coded_prompts_v2.json',
        'clip_friendly_blur': 'configs/prompts/llm_text_prompts_v2_clip_friendly.json',
    }
    
    results = evaluate_best_combo(dataset, clip_classifier, prompt_configs)
    
    # Summary
    log("\n" + "="*70)
    log("RESULTS SUMMARY")
    log("="*70)
    log(f"\n{'Config':<30} {'Accuracy':>12} {'vs Previous':>15}")
    log("-"*60)
    
    # Previous results without blur
    previous = {
        'fewshot_blur': 0.389,  # was 38.9% without blur
        'hardcoded_v2_blur': 0.356,  # was 35.6% without blur
        'clip_friendly_blur': 0.339,  # was 33.9% without blur
    }
    
    for name, res in sorted(results.items(), key=lambda x: -x[1]['accuracy']):
        acc = res['accuracy']
        prev = previous.get(name, 0)
        delta = acc - prev
        delta_str = f"+{delta:.1%}" if delta > 0 else f"{delta:.1%}"
        log(f"{name:<30} {acc:>11.2%} {delta_str:>15}")
    
    # Per-class for best
    if results:
        best_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
        best = results[best_name]
        log(f"\nBest config: {best_name}")
        log(f"Per-class breakdown:")
        for cn, acc in best['per_class'].items():
            log(f"  {cn}: {acc:.2%}")


if __name__ == '__main__':
    main()
