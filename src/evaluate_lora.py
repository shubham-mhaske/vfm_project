"""
Evaluate SAM2 with LoRA adapters on the BCSS test set.

This script loads a LoRA-adapted SAM2 model and evaluates it using the full
SAM2ImagePredictor with proper prompts (box/point), NOT the simplified 
backbone-only forward pass used during training.

This ensures fair comparison with the zeroshot baseline (0.517 Dice).

Usage:
    python src/evaluate_lora.py \
        --lora_weights finetune_logs/lora/lora_r8_XXXXX/best_lora_weights.pt \
        --output_dir results/lora_r8_eval
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime

# Add project paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sam2_root = os.path.join(project_root, 'sam2')
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if sam2_root not in sys.path:
    sys.path.insert(0, sam2_root)

print("[BOOT] Starting evaluate_lora.py", flush=True)
import torch
import numpy as np
from tqdm import tqdm

from src.dataset import BCSSDataset
from src.sam_segmentation import calculate_metrics, get_prompts_from_mask, get_predicted_mask_from_prompts
from src.device_utils import get_device
from src.lora_adapter import apply_lora_to_sam2


def load_lora_sam2(model_cfg: str, base_checkpoint: str, lora_weights: str, 
                   lora_rank: int, device: str) -> 'SAM2ImagePredictor':
    """
    Load SAM2 with LoRA weights for evaluation.
    
    Args:
        model_cfg: Path to SAM2 config (e.g., 'configs/sam2.1/sam2.1_hiera_l.yaml')
        base_checkpoint: Path to base SAM2 checkpoint
        lora_weights: Path to saved LoRA weights (.pt file)
        lora_rank: Rank used when training LoRA (must match!)
        device: Device to load model on
    
    Returns:
        SAM2ImagePredictor ready for inference
    """
    from training.utils.train_utils import register_omegaconf_resolvers
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    
    print(f"[LoRA-Eval] Loading base SAM2 model...")
    print(f"[LoRA-Eval]   Config: {model_cfg}")
    print(f"[LoRA-Eval]   Checkpoint: {base_checkpoint}")
    
    # Register OmegaConf resolvers
    register_omegaconf_resolvers()
    
    # Change to sam2 directory for Hydra config resolution
    orig_cwd = os.getcwd()
    os.chdir(sam2_root)
    
    try:
        # Build base SAM2 model with pretrained weights
        model = build_sam2(model_cfg, ckpt_path=base_checkpoint, device='cpu')
    finally:
        os.chdir(orig_cwd)
    
    # Apply LoRA adapters (same architecture as training)
    print(f"[LoRA-Eval] Applying LoRA adapters (rank={lora_rank})...")
    lora_model = apply_lora_to_sam2(
        sam2_model=model,
        r=lora_rank,
        alpha=float(lora_rank),
        dropout=0.0,  # No dropout during inference
        target_modules='image_encoder',
        trainable_output_head=True,
    )
    
    # Load the trained LoRA weights
    print(f"[LoRA-Eval] Loading LoRA weights from: {lora_weights}")
    lora_state_dict = torch.load(lora_weights, map_location='cpu', weights_only=True)
    
    # Update model with LoRA weights
    current_state = lora_model.sam2_model.state_dict()
    current_state.update(lora_state_dict)
    lora_model.sam2_model.load_state_dict(current_state)
    
    print(f"[LoRA-Eval] Loaded {len(lora_state_dict)} LoRA weight tensors")
    
    # Move to device and set eval mode
    lora_model = lora_model.to(device)
    lora_model.eval()
    
    # Create predictor from the adapted model
    predictor = SAM2ImagePredictor(lora_model.sam2_model)
    
    return predictor


def evaluate_lora(args):
    """Run full SAM2 evaluation with LoRA weights on test set."""
    
    # Setup paths
    image_dir = os.path.join(project_root, args.image_dir)
    mask_dir = os.path.join(project_root, args.mask_dir)
    lora_weights = os.path.join(project_root, args.lora_weights) if not os.path.isabs(args.lora_weights) else args.lora_weights
    base_checkpoint = os.path.join(project_root, args.sam_checkpoint) if not os.path.isabs(args.sam_checkpoint) else args.sam_checkpoint
    
    if args.output_dir:
        output_dir = os.path.join(project_root, args.output_dir)
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = None
    
    # Load dataset
    print(f"\n[LoRA-Eval] Loading {args.split} dataset...")
    dataset = BCSSDataset(image_dir=image_dir, mask_dir=mask_dir, split=args.split)
    print(f"[LoRA-Eval] Dataset size: {len(dataset)}")
    
    # Get device
    device = get_device(force_cpu=args.force_cpu)
    print(f"[LoRA-Eval] Using device: {device}")
    
    # Load LoRA model
    predictor = load_lora_sam2(
        model_cfg=args.sam_model_cfg,
        base_checkpoint=base_checkpoint,
        lora_weights=lora_weights,
        lora_rank=args.lora_rank,
        device=device
    )
    
    # Evaluation loop
    num_samples = len(dataset) if args.max_samples is None else min(args.max_samples, len(dataset))
    
    class_names = dataset.class_names
    class_dice_scores = {cid: [] for cid in class_names if cid != 0}
    class_iou_scores = {cid: [] for cid in class_names if cid != 0}
    sample_results = []
    
    print(f"\n[LoRA-Eval] Evaluating {num_samples} samples with prompt_type='{args.prompt_type}'...")
    eval_start = time.time()
    
    for i in tqdm(range(num_samples), disable=not args.tqdm):
        sample = dataset[i]
        image = sample['image_np']
        gt_mask = sample['mask'].numpy()
        unique_classes = np.unique(gt_mask)
        
        # Set image for predictor
        predictor.set_image(image)
        
        for class_id in unique_classes:
            if class_id == 0:
                continue
            
            class_name = class_names.get(class_id)
            if not class_name:
                continue
            
            # Create binary ground truth mask for this class
            binary_gt_mask = (gt_mask == class_id).astype(np.uint8)
            
            # Get prompts from ground truth (simulating oracle prompts)
            prompts = get_prompts_from_mask(binary_gt_mask)
            
            if args.prompt_type not in prompts:
                continue
            
            # Get prediction using full SAM2 predictor
            masks_logits, _, _ = get_predicted_mask_from_prompts(
                predictor, image, prompts, 
                prompt_type=args.prompt_type,
                use_neg_points=args.use_neg_points
            )
            
            # Threshold to get binary prediction
            predicted_mask = (masks_logits > 0.5).astype(np.uint8)
            
            # Calculate metrics
            dice, iou = calculate_metrics(predicted_mask, binary_gt_mask)
            
            class_dice_scores[class_id].append(dice)
            class_iou_scores[class_id].append(iou)
            sample_results.append({
                'sample_idx': i,
                'class_id': class_id,
                'class_name': class_name,
                'dice': dice,
                'iou': iou,
                'filename': dataset.image_files[i]
            })
    
    eval_time = time.time() - eval_start
    
    # Compute per-class and overall metrics
    results = {
        'meta': {
            'lora_weights': args.lora_weights,
            'lora_rank': args.lora_rank,
            'prompt_type': args.prompt_type,
            'split': args.split,
            'num_samples': num_samples,
            'eval_time_seconds': eval_time,
            'timestamp': datetime.now().isoformat()
        },
        'per_class': {},
        'overall': {}
    }
    
    all_dice = []
    all_iou = []
    
    print("\n" + "="*60)
    print("PER-CLASS RESULTS")
    print("="*60)
    
    for cid, cname in class_names.items():
        if cid == 0:
            continue
        
        dice_scores = class_dice_scores.get(cid, [])
        iou_scores = class_iou_scores.get(cid, [])
        
        if dice_scores:
            mean_dice = np.mean(dice_scores)
            std_dice = np.std(dice_scores)
            mean_iou = np.mean(iou_scores)
            std_iou = np.std(iou_scores)
            
            results['per_class'][cname] = {
                'dice_mean': float(mean_dice),
                'dice_std': float(std_dice),
                'iou_mean': float(mean_iou),
                'iou_std': float(std_iou),
                'count': len(dice_scores)
            }
            
            all_dice.extend(dice_scores)
            all_iou.extend(iou_scores)
            
            print(f"{cname:20s}: Dice={mean_dice:.4f}±{std_dice:.4f}, IoU={mean_iou:.4f}±{std_iou:.4f} (n={len(dice_scores)})")
        else:
            print(f"{cname:20s}: No samples")
    
    # Overall metrics
    if all_dice:
        overall_dice = np.mean(all_dice)
        overall_iou = np.mean(all_iou)
        
        results['overall'] = {
            'dice_mean': float(overall_dice),
            'dice_std': float(np.std(all_dice)),
            'iou_mean': float(overall_iou),
            'iou_std': float(np.std(all_iou)),
            'total_samples': len(all_dice)
        }
        
        print("="*60)
        print(f"OVERALL: Dice={overall_dice:.4f}, IoU={overall_iou:.4f}")
        print("="*60)
        
        # Comparison with baseline
        zeroshot_baseline = 0.517
        improvement = overall_dice - zeroshot_baseline
        pct_change = 100 * improvement / zeroshot_baseline
        
        print(f"\n📊 COMPARISON WITH ZEROSHOT BASELINE:")
        print(f"   Zeroshot SAM2:  {zeroshot_baseline:.4f} Dice")
        print(f"   LoRA SAM2:      {overall_dice:.4f} Dice")
        print(f"   Difference:     {improvement:+.4f} ({pct_change:+.1f}%)")
        
        if improvement > 0:
            print(f"   ✅ LoRA IMPROVES over zeroshot!")
        else:
            print(f"   ❌ LoRA does NOT improve over zeroshot")
        
        results['comparison'] = {
            'zeroshot_baseline': zeroshot_baseline,
            'improvement': float(improvement),
            'pct_change': float(pct_change)
        }
    
    # Save results
    if output_dir:
        metrics_path = os.path.join(output_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {metrics_path}")
        
        # Save detailed sample results
        samples_path = os.path.join(output_dir, 'sample_results.json')
        with open(samples_path, 'w') as f:
            json.dump(sample_results, f, indent=2)
        print(f"💾 Sample details saved to: {samples_path}")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate SAM2 with LoRA on BCSS test set')
    
    # LoRA weights (REQUIRED)
    parser.add_argument('--lora_weights', type=str, required=True,
                        help='Path to saved LoRA weights (.pt file)')
    parser.add_argument('--lora_rank', type=int, default=8,
                        help='Rank used when training LoRA (must match training!)')
    
    # Model configuration
    parser.add_argument('--sam_model_cfg', type=str,
                        default='configs/sam2.1/sam2.1_hiera_l.yaml',
                        help='SAM2 config (relative to sam2 package)')
    parser.add_argument('--sam_checkpoint', type=str,
                        default='sam2/checkpoints/sam2.1_hiera_large.pt',
                        help='Path to base SAM2 checkpoint')
    
    # Data configuration
    parser.add_argument('--image_dir', type=str, default='data/bcss/images',
                        help='Path to image directory')
    parser.add_argument('--mask_dir', type=str, default='data/bcss/masks',
                        help='Path to mask directory')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate')
    
    # Prompt configuration
    parser.add_argument('--prompt_type', type=str, default='box',
                        choices=['centroid', 'multi_point', 'box'],
                        help='Type of prompt to use')
    parser.add_argument('--use_neg_points', action='store_true',
                        help='Use negative points in addition to positive prompts')
    
    # Output configuration
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save results')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Limit number of samples (for quick testing)')
    parser.add_argument('--tqdm', action='store_true',
                        help='Show progress bar')
    parser.add_argument('--force_cpu', action='store_true',
                        help='Force CPU usage')
    
    args = parser.parse_args()
    evaluate_lora(args)
