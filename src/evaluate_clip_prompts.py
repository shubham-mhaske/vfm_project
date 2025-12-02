#!/usr/bin/env python3
"""
Comprehensive CLIP Prompt Evaluation
Tests all CLIP prompt configurations with the best SAM2 settings:
- SAM2: Box + Negative Points + TTA (best segmentation: 0.566 Dice)
- CLIP: All prompt files (hard_coded, llm_text, llm_multimodal)
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
from datetime import datetime

from dataset import BCSSDataset
from sam_segmentation import get_sam2_predictor, get_prompts_from_mask, calculate_metrics
from clip_classification import CLIPClassifier, crop_region_from_mask, load_prompts_from_json, get_clip_model_path
from device_utils import get_device
from tta_utils import predict_with_tta
from iterative_refinement import predict_with_tta_and_refinement
from training.utils.train_utils import register_omegaconf_resolvers


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def evaluate_clip_prompts(
    predictor, clip_classifier, clip_prompts, dataset,
    use_tta=True, use_refinement=True
):
    """
    Evaluate CLIP classification with a specific prompt set.
    Uses best SAM2 config (Box + Neg + TTA + Refinement) for segmentation.
    """
    class_names = dataset.class_names
    
    # Storage
    class_dice = {cid: [] for cid in class_names if cid != 0}
    class_iou = {cid: [] for cid in class_names if cid != 0}
    class_correct = {cid: 0 for cid in class_names if cid != 0}
    class_total = {cid: 0 for cid in class_names if cid != 0}
    
    from sam_segmentation import get_predicted_mask_from_prompts
    
    for i in tqdm(range(len(dataset)), desc="Evaluating"):
        sample = dataset[i]
        image = sample['image_np']
        gt_mask = sample['mask'].numpy()
        
        for class_id in np.unique(gt_mask):
            if class_id == 0:
                continue
            
            class_name = class_names.get(class_id)
            if not class_name:
                continue
            
            # Skip if class not in CLIP prompts
            if class_name not in clip_prompts:
                continue
            
            binary_gt = (gt_mask == class_id).astype(np.uint8)
            prompts = get_prompts_from_mask(binary_gt)
            
            if 'box' not in prompts:
                continue
            
            # Best SAM2 config: Box + Neg + TTA + Refinement
            if use_tta and use_refinement:
                pred_mask = predict_with_tta_and_refinement(
                    predictor, image, prompts, 'box', True,
                    num_augmentations=4, num_refinement_iterations=2
                )
            elif use_tta:
                pred_mask = predict_with_tta(predictor, image, prompts, 'box', True)
            else:
                predictor.set_image(image)
                mask_logits, _, _ = get_predicted_mask_from_prompts(
                    predictor, image, prompts, 'box', True
                )
                pred_mask = (mask_logits > 0.5).astype(np.uint8)
            
            # Segmentation metrics
            dice, iou = calculate_metrics(pred_mask, binary_gt)
            class_dice[class_id].append(dice)
            class_iou[class_id].append(iou)
            
            # CLIP classification
            if pred_mask.sum() > 0:
                cropped = crop_region_from_mask(image, pred_mask)
                if cropped:
                    pred_class = clip_classifier.classify_region(cropped, clip_prompts)
                    class_total[class_id] += 1
                    if pred_class == class_name:
                        class_correct[class_id] += 1
    
    # Aggregate
    results = {'per_class': {}}
    
    for cid, cname in class_names.items():
        if cid == 0:
            continue
        if cname not in clip_prompts:
            continue
            
        results['per_class'][cname] = {
            'dice': float(np.mean(class_dice[cid])) if class_dice[cid] else 0,
            'iou': float(np.mean(class_iou[cid])) if class_iou[cid] else 0,
            'clip_acc': float(class_correct[cid] / class_total[cid]) if class_total[cid] > 0 else 0,
            'correct': class_correct[cid],
            'total': class_total[cid]
        }
    
    all_dice = [s for scores in class_dice.values() for s in scores]
    all_iou = [s for scores in class_iou.values() for s in scores]
    
    total_correct = sum(class_correct.values())
    total_samples = sum(class_total.values())
    
    results['overall'] = {
        'dice': float(np.mean(all_dice)) if all_dice else 0,
        'iou': float(np.mean(all_iou)) if all_iou else 0,
        'clip_acc': float(total_correct / total_samples) if total_samples > 0 else 0,
        'correct': total_correct,
        'total': total_samples
    }
    
    return results


def get_dual_gpu_devices():
    """Get separate devices for SAM2 and CLIP when 2 GPUs available."""
    import torch
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus >= 2:
            log(f"Multi-GPU mode: {num_gpus} GPUs detected")
            log(f"  GPU 0 ({torch.cuda.get_device_name(0)}): SAM2")
            log(f"  GPU 1 ({torch.cuda.get_device_name(1)}): CLIP")
            return "cuda:0", "cuda:1"
        else:
            log(f"Single GPU mode: {torch.cuda.get_device_name(0)}")
            return "cuda:0", "cuda:0"
    return "cpu", "cpu"


def main():
    parser = argparse.ArgumentParser(description="Evaluate all CLIP prompt configurations")
    parser.add_argument('--model_cfg', default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument('--checkpoint', default="sam2/checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument('--split', default='test')
    parser.add_argument('--output_dir', default='results/clip_prompts_comparison')
    parser.add_argument('--use_tta', action='store_true', default=True)
    parser.add_argument('--use_refinement', action='store_true', default=True)
    args = parser.parse_args()
    
    log("="*80)
    log("CLIP Prompt Comparison Evaluation")
    log("SAM2 Config: Box + Negative Points + TTA + Refinement (Best)")
    log("="*80)
    
    # Get separate devices for SAM2 and CLIP
    sam_device, clip_device = get_dual_gpu_devices()
    log(f"SAM2 device: {sam_device}")
    log(f"CLIP device: {clip_device}")
    
    register_omegaconf_resolvers()
    
    # Load dataset
    log("Loading dataset...")
    dataset = BCSSDataset('data/bcss/images', 'data/bcss/masks', split=args.split)
    log(f"Loaded {len(dataset)} samples")
    
    # Load SAM2 on GPU 0
    log("Loading SAM2...")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sam2_root = os.path.join(project_root, 'sam2')
    checkpoint = os.path.join(project_root, args.checkpoint)
    
    orig_cwd = os.getcwd()
    os.chdir(sam2_root)
    try:
        predictor = get_sam2_predictor("configs/sam2.1/sam2.1_hiera_l.yaml", checkpoint, sam_device)
    finally:
        os.chdir(orig_cwd)
    log("SAM2 loaded!")
    
    # Load CLIP on GPU 1
    log("Loading CLIP...")
    clip_classifier = CLIPClassifier(device=clip_device)
    log("CLIP loaded!")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # All CLIP prompt configurations to test
    prompt_configs = [
        # Exp1: Hard-coded prompts
        {'name': 'exp1_hardcoded_v1', 'file': 'configs/prompts/hard_coded_prompts.json'},
        {'name': 'exp1_hardcoded_v2', 'file': 'configs/prompts/hard_coded_prompts_v2.json'},
        
        # Exp3: LLM text prompts
        {'name': 'exp3_llm_text_v1_jargon', 'file': 'configs/prompts/llm_text_prompts_v1_gemini_pro_latest.json'},
        {'name': 'exp3_llm_text_v2_clip_friendly', 'file': 'configs/prompts/llm_text_prompts_v2_clip_friendly.json'},
        {'name': 'exp3_llm_text_v3_fewshot', 'file': 'configs/prompts/llm_text_prompts_v3_fewshot.json'},
        
        # Exp5: LLM multimodal prompts
        {'name': 'exp5_llm_multimodal_v1', 'file': 'configs/prompts/llm_multimodal_prompts_v1_gemini_2.5_flash.json'},
        {'name': 'exp5_llm_multimodal_v2_clip_friendly', 'file': 'configs/prompts/llm_multimodal_prompts_v2_clip_friendly.json'},
        {'name': 'exp5_llm_multimodal_v3_fewshot', 'file': 'configs/prompts/llm_multimodal_prompts_v3_fewshot.json'},
    ]
    
    all_results = {}
    
    for i, cfg in enumerate(prompt_configs):
        prompt_path = os.path.join(project_root, cfg['file'])
        
        if not os.path.exists(prompt_path):
            log(f"\n[{i+1}/{len(prompt_configs)}] SKIP: {cfg['name']} (file not found)")
            continue
        
        log(f"\n>>> [{i+1}/{len(prompt_configs)}] Testing: {cfg['name']}")
        
        try:
            clip_prompts = load_prompts_from_json(prompt_path)
            log(f"    Classes in prompts: {list(clip_prompts.keys())}")
        except Exception as e:
            log(f"    ERROR loading prompts: {e}")
            continue
        
        start = time.time()
        
        results = evaluate_clip_prompts(
            predictor, clip_classifier, clip_prompts, dataset,
            use_tta=args.use_tta, use_refinement=args.use_refinement
        )
        
        elapsed = time.time() - start
        results['time_seconds'] = elapsed
        results['prompt_file'] = cfg['file']
        all_results[cfg['name']] = results
        
        ov = results['overall']
        log(f"    Dice: {ov['dice']:.4f} | IoU: {ov['iou']:.4f} | "
            f"CLIP Acc: {ov['clip_acc']:.2%} ({ov['correct']}/{ov['total']}) | Time: {elapsed:.1f}s")
    
    # Summary
    log("\n" + "="*80)
    log("RESULTS SUMMARY")
    log("="*80)
    log(f"{'Prompt Config':<35} {'Dice':>8} {'IoU':>8} {'CLIP Acc':>10} {'Correct':>10}")
    log("-"*75)
    
    # Sort by CLIP accuracy
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]['overall']['clip_acc'], reverse=True)
    
    for name, res in sorted_results:
        ov = res['overall']
        log(f"{name:<35} {ov['dice']:>8.4f} {ov['iou']:>8.4f} {ov['clip_acc']:>9.2%} {ov['correct']:>5}/{ov['total']:<4}")
    
    # Best config per-class breakdown
    if sorted_results:
        best_name, best = sorted_results[0]
        log(f"\n{'='*80}")
        log(f"BEST CLIP PROMPTS: {best_name}")
        log(f"Prompt file: {best['prompt_file']}")
        log("="*80)
        log(f"{'Class':<15} {'Dice':>8} {'IoU':>8} {'CLIP Acc':>10} {'Correct':>10}")
        log("-"*55)
        
        for cname in ['tumor', 'stroma', 'lymphocyte', 'necrosis', 'blood_vessel']:
            if cname in best['per_class']:
                pc = best['per_class'][cname]
                log(f"{cname:<15} {pc['dice']:>8.4f} {pc['iou']:>8.4f} {pc['clip_acc']:>9.2%} {pc['correct']:>5}/{pc['total']:<4}")
    
    # Save results
    output_path = os.path.join(args.output_dir, 'clip_prompts_comparison.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    log(f"\nResults saved to: {output_path}")
    
    # Save summary CSV
    csv_path = os.path.join(args.output_dir, 'summary.csv')
    with open(csv_path, 'w') as f:
        f.write("Prompt_Config,Dice,IoU,CLIP_Accuracy,Correct,Total,Time_s\n")
        for name, res in sorted_results:
            ov = res['overall']
            f.write(f"{name},{ov['dice']:.4f},{ov['iou']:.4f},{ov['clip_acc']:.4f},"
                    f"{ov['correct']},{ov['total']},{res['time_seconds']:.1f}\n")
    log(f"Summary CSV saved to: {csv_path}")


if __name__ == '__main__':
    main()
