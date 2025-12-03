import sys
import os
# Add project root to path for local imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
# Add sam2 repo root to path BEFORE project root for training module access
sam2_root = os.path.join(project_root, 'sam2')
sys.path.insert(0, sam2_root)

print("[BOOT] Starting evaluate_segmentation.py", flush=True)
import time
import json
boot_t0 = time.time()
print("[BOOT] Importing torch...", flush=True)
import torch
print("[BOOT] Importing numpy...", flush=True)
import numpy as np
print("[BOOT] Importing tqdm...", flush=True)
from tqdm import tqdm
print("[BOOT] Importing dataset module...", flush=True)
from dataset import BCSSDataset
print("[BOOT] Importing sam_segmentation helpers...", flush=True)
from sam_segmentation import get_sam2_predictor, get_predicted_mask, calculate_metrics
print("[BOOT] Importing argparse...", flush=True)
import argparse
print("[BOOT] Importing device_utils...", flush=True)
from device_utils import get_device
print("[BOOT] Importing training utils (OmegaConf resolvers)...", flush=True)
from training.utils.train_utils import register_omegaconf_resolvers
print(f"[BOOT] Imports done in {time.time()-boot_t0:.2f}s", flush=True)


def is_pathsam_ctranspath_checkpoint(state_dict: dict) -> bool:
    """
    Detect if checkpoint contains Path-SAM2 + CTransPath weights.
    
    Returns True if the checkpoint has keys like:
        image_encoder.sam_encoder.trunk.*
        image_encoder.ctranspath_encoder.*
        image_encoder.dimension_alignment.*
    """
    pathsam_indicators = [
        'image_encoder.sam_encoder.',
        'image_encoder.ctranspath_encoder.',
        'image_encoder.dimension_alignment.'
    ]
    for key in state_dict.keys():
        for indicator in pathsam_indicators:
            if key.startswith(indicator):
                return True
    return False


def load_pathsam_ctranspath_model(model_cfg: str, checkpoint_path: str, 
                                   ctranspath_ckpt: str, device: str,
                                   fusion_type: str = 'concat'):
    """
    Load a Path-SAM2 + CTransPath model with the correct architecture.
    
    Args:
        model_cfg: Path to SAM2 model config
        checkpoint_path: Path to finetuned Path-SAM2 checkpoint
        ctranspath_ckpt: Path to CTransPath pretrained weights
        device: Device to load model on
        fusion_type: Fusion type used during training ('concat' or 'attention')
    
    Returns:
        SAM2ImagePredictor with Path-SAM2 + CTransPath encoder
    """
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from src.ctranspath_encoder import PathSAM2CTransPathEncoder
    
    print(f"[PathSAM2] Building base SAM2 model from config: {model_cfg}")
    
    # Build base SAM2 model (without loading checkpoint - we'll load our finetuned weights)
    model = build_sam2(model_cfg, ckpt_path=None, device=device)
    
    # Wrap the image encoder with PathSAM2CTransPathEncoder
    print(f"[PathSAM2] Integrating CTransPath encoder...")
    print(f"[PathSAM2]   CTransPath checkpoint: {ctranspath_ckpt}")
    print(f"[PathSAM2]   Fusion type: {fusion_type}")
    
    model.image_encoder = PathSAM2CTransPathEncoder(
        sam_encoder=model.image_encoder,
        ctranspath_checkpoint=ctranspath_ckpt,
        freeze_sam=True,
        freeze_ctranspath=True,
        fusion_type=fusion_type
    )
    
    # Load the finetuned checkpoint
    print(f"[PathSAM2] Loading finetuned weights from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('model', ckpt)
    
    # Load state dict with strict=False to handle any minor mismatches
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    
    if missing:
        print(f"[PathSAM2] Warning: {len(missing)} missing keys (expected for frozen components)")
    if unexpected:
        print(f"[PathSAM2] Warning: {len(unexpected)} unexpected keys")
    
    print(f"[PathSAM2] Model loaded successfully!")
    model = model.to(device)
    model.eval()
    
    return SAM2ImagePredictor(model)


def convert_pathsam_to_standard_sam(state_dict: dict) -> dict:
    """
    Convert Path-SAM2 checkpoint keys to standard SAM2 format.
    
    This allows loading Path-SAM2 checkpoints into vanilla SAM2 by:
    1. Remapping image_encoder.sam_encoder.* -> image_encoder.*
    2. Dropping CTransPath and fusion layer weights
    
    Note: This loses the CTransPath features! Use load_pathsam_ctranspath_model()
    for full Path-SAM2 inference.
    """
    new_state = {}
    dropped_keys = 0
    
    for key, value in state_dict.items():
        if key.startswith('image_encoder.sam_encoder.'):
            # Remap: image_encoder.sam_encoder.trunk.* -> image_encoder.trunk.*
            new_key = key.replace('image_encoder.sam_encoder.', 'image_encoder.')
            new_state[new_key] = value
        elif key.startswith('image_encoder.ctranspath_encoder.'):
            # Drop CTransPath weights
            dropped_keys += 1
            continue
        elif key.startswith('image_encoder.dimension_alignment.'):
            # Drop fusion layer weights
            dropped_keys += 1
            continue
        else:
            # Keep other keys as-is (mask decoder, memory modules, etc.)
            new_state[key] = value
    
    print(f"[Checkpoint] Converted Path-SAM2 -> SAM2: kept {len(new_state)} keys, dropped {dropped_keys} CTransPath/fusion keys")
    return new_state

def evaluate_segmentation(args):
    """Runs the SAM 2 segmentation and evaluation on the full test set (or a subset)."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sam2_root = os.path.join(project_root, 'sam2')
    image_dir = os.path.join(project_root, args.image_dir)
    mask_dir = os.path.join(project_root, args.mask_dir)

    output_dir = None
    if getattr(args, 'output_dir', None) and not getattr(args, 'print_only', False):
        output_dir = os.path.join(project_root, args.output_dir)
        os.makedirs(output_dir, exist_ok=True)

    bcss_dataset = BCSSDataset(image_dir=image_dir, mask_dir=mask_dir, split=args.split)
    device = get_device(force_cpu=args.force_cpu) if hasattr(args, 'force_cpu') else get_device()
    print(f"Using device: {device}", flush=True)
    register_omegaconf_resolvers()

    checkpoint_path = os.path.join(project_root, args.checkpoint)
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Determine checkpoint type and load appropriately
    state_dict = ckpt.get('model', ckpt) if isinstance(ckpt, dict) else ckpt
    
    # Change to sam2 directory for Hydra config resolution
    orig_cwd = os.getcwd()
    os.chdir(sam2_root)
    
    try:
        if isinstance(state_dict, dict) and is_pathsam_ctranspath_checkpoint(state_dict):
            # Path-SAM2 + CTransPath checkpoint detected
            print("\n" + "="*60)
            print("[Checkpoint] Detected Path-SAM2 + CTransPath checkpoint!")
            print("="*60)
            
            if args.use_pathsam:
                # Load full Path-SAM2 model with CTransPath fusion
                ctranspath_ckpt = os.path.join(project_root, args.ctranspath_checkpoint)
                if not os.path.exists(ctranspath_ckpt):
                    raise FileNotFoundError(
                        f"CTransPath checkpoint not found at: {ctranspath_ckpt}\n"
                        f"Please provide --ctranspath_checkpoint or use --no_pathsam to load as standard SAM2"
                    )
                predictor = load_pathsam_ctranspath_model(
                    args.model_cfg, 
                    checkpoint_path, 
                    ctranspath_ckpt, 
                    device,
                    fusion_type=args.fusion_type
                )
            else:
                # Convert to standard SAM2 format (loses CTransPath features)
                print("[Checkpoint] Converting to standard SAM2 format (--no_pathsam mode)")
                print("[Checkpoint] WARNING: CTransPath fusion features will be ignored!")
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor
                model = build_sam2(args.model_cfg, ckpt_path=None, device=device)
                converted_state = convert_pathsam_to_standard_sam(state_dict)
                model.load_state_dict(converted_state, strict=False)
                predictor = SAM2ImagePredictor(model)
        elif 'model' in ckpt and isinstance(ckpt['model'], dict):
            # Standard finetuned SAM2 checkpoint
            print("[Checkpoint] Standard finetuned SAM2 checkpoint detected")
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            model = build_sam2(args.model_cfg, ckpt_path=None, device=device)
            model.load_state_dict(ckpt['model'], strict=False)
            predictor = SAM2ImagePredictor(model)
        else:
            # Original SAM2 checkpoint format
            print("[Checkpoint] Original SAM2 checkpoint format detected")
            predictor = get_sam2_predictor(args.model_cfg, checkpoint_path, device)
    finally:
        # Restore original working directory
        os.chdir(orig_cwd)

    num_samples = len(bcss_dataset) if args.max_samples is None else min(args.max_samples, len(bcss_dataset))
    
    class_thresholds = {}
    if args.threshold_config and os.path.exists(args.threshold_config):
        with open(args.threshold_config, 'r') as f:
            class_thresholds = json.load(f)
        print(f"Loaded custom thresholds from: {args.threshold_config}")

    class_names = bcss_dataset.class_names
    class_dice_scores = {cid: [] for cid in class_names if cid != 0}
    class_iou_scores = {cid: [] for cid in class_names if cid != 0}  # Added IoU tracking
    sample_results = []

    loop_start = time.time()
    for i in tqdm(range(num_samples), disable=not args.tqdm):
        sample = bcss_dataset[i]
        image = sample['image_np']
        gt_mask = sample['mask'].numpy()
        unique_classes = np.unique(gt_mask)

        predictor.set_image(image)

        for class_id in unique_classes:
            if class_id == 0: continue
            
            class_name = class_names.get(class_id)
            if not class_name: continue

            binary_gt_mask = (gt_mask == class_id).astype(np.uint8)
            
            from src.sam_segmentation import get_prompts_from_mask
            prompts = get_prompts_from_mask(binary_gt_mask)

            if args.prompt_type not in prompts:
                continue

            if args.use_tta:
                from src.tta_utils import predict_with_tta
                predicted_mask = predict_with_tta(predictor, image, prompts, prompt_type=args.prompt_type, use_neg_points=args.use_neg_points)
            else:
                from src.sam_segmentation import get_predicted_mask_from_prompts
                masks_logits, _, _ = get_predicted_mask_from_prompts(predictor, image, prompts, prompt_type=args.prompt_type, use_neg_points=args.use_neg_points)
                threshold = class_thresholds.get(class_name, 0.5)
                predicted_mask = (masks_logits > threshold).astype(np.uint8)
            
            dice, iou = calculate_metrics(predicted_mask, binary_gt_mask)
            class_dice_scores[class_id].append(dice)
            class_iou_scores[class_id].append(iou)  # Store IoU
            sample_results.append({
                'sample_idx': i, 
                'class_id': class_id, 
                'dice': float(dice), 
                'iou': float(iou),  # Added IoU
                'image_path': bcss_dataset.image_files[i]
            })

    # --- Metrics Calculation and Reporting ---
    results = {}
    for cid, cname in class_names.items():
        if cid == 0: continue
        dice_scores = class_dice_scores.get(cid, [])
        iou_scores = class_iou_scores.get(cid, [])
        results[cname] = {
            'dice': float(np.mean(dice_scores)) if dice_scores else 0, 
            'dice_std': float(np.std(dice_scores)) if dice_scores else 0,
            'iou': float(np.mean(iou_scores)) if iou_scores else 0,
            'iou_std': float(np.std(iou_scores)) if iou_scores else 0,
            'count': len(dice_scores)
        }
    
    all_dice = [s for scores in class_dice_scores.values() for s in scores]
    all_iou = [s for scores in class_iou_scores.values() for s in scores]
    results['overall'] = {
        'dice': float(np.mean(all_dice)) if all_dice else 0,
        'dice_std': float(np.std(all_dice)) if all_dice else 0,
        'iou': float(np.mean(all_iou)) if all_iou else 0,
        'iou_std': float(np.std(all_iou)) if all_iou else 0,
        'total_samples': len(all_dice)
    }
    
    print("\n--- Per-Class Metrics (Dice | IoU) ---")
    for cname, data in results.items():
        if cname != 'overall':
            print(f"{cname:15s}: {data['dice']:.4f} ± {data['dice_std']:.4f} | {data['iou']:.4f} ± {data['iou_std']:.4f} (n={data['count']})")
    print(f"-----------------------------------------")
    print(f"Overall:        Dice={results['overall']['dice']:.4f} | IoU={results['overall']['iou']:.4f}")

    # --- Save Visualizations ---
    if args.save_predictions and output_dir is not None:
        vis_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        sample_results.sort(key=lambda x: x['dice'])
        
        best_samples = sample_results[-3:]
        worst_samples = sample_results[:3]
        
        from sam_segmentation import save_prediction_visualization
        for res in best_samples + worst_samples:
            sample = bcss_dataset[res['sample_idx']]
            image = sample['image_np']
            gt_mask = (sample['mask'].numpy() == res['class_id']).astype(np.uint8)
            # Re-predict to get the mask for visualization
            prompts = get_prompts_from_mask(gt_mask)
            predicted_mask, _, _ = get_predicted_mask_from_prompts(predictor, image, prompts, args.prompt_type, args.use_neg_points)
            
            fname = os.path.basename(res['image_path'])
            vis_path = os.path.join(vis_dir, f"dice_{res['dice']:.4f}_{class_names[res['class_id']]}_{fname}")
            save_prediction_visualization(image, gt_mask, (predicted_mask > class_thresholds.get(class_names[res['class_id']], 0.5)), res['dice'], vis_path)
        print(f"Saved visualizations to {vis_dir}")

    # --- Save Metrics ---
    if output_dir is not None and not getattr(args, 'print_only', False):
        metrics_path = os.path.join(output_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Metrics saved to {metrics_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate SAM 2 segmentation.")
    parser.add_argument('--checkpoint', type=str, required=True, 
                        help='Path to SAM checkpoint (relative to project root)')
    parser.add_argument('--model_cfg', type=str, 
                        default='sam2/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml',
                        help='Path to SAM2 model config for inference (relative to project root). '
                             'Use inference configs (e.g., sam2.1_hiera_b+.yaml) not training configs.')
    parser.add_argument('--image_dir', type=str, 
                        default='data/bcss/images',
                        help='Path to image directory (relative to project root)')
    parser.add_argument('--mask_dir', type=str, 
                        default='data/bcss/masks',
                        help='Path to mask directory (relative to project root)')
    parser.add_argument('--split', type=str, 
                        default='test',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate on')
    parser.add_argument('--prompt_type', type=str,
                        default='centroid',
                        choices=['centroid', 'multi_point', 'box'],
                        help='Type of prompt to use for SAM2 (centroid, multi_point, or box)')
    parser.add_argument('--use_neg_points', action='store_true',
                        help='Use negative points (outside the mask) in addition to positive prompts')
    parser.add_argument('--force_cpu', action='store_true',
                        help='Force CPU usage even if GPU is available')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose step-by-step logging')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Limit number of samples evaluated (for debugging)')
    parser.add_argument('--tqdm', action='store_true',
                        help='Enable tqdm progress bar (disabled by default in Slurm logs)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory (relative to project root) to save metrics.json')
    parser.add_argument('--use_tta', action='store_true',
                        help='Enable test-time augmentation for more robust predictions.')
    parser.add_argument('--threshold_config', type=str, default=None,
                        help='Path to a JSON file with per-class confidence thresholds.')
    parser.add_argument('--print_only', action='store_true',
                        help='Run evaluation and print metrics without writing any files (ignores output_dir).')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save visualizations of predictions to the output directory.')
    
    # Path-SAM2 + CTransPath options
    parser.add_argument('--use_pathsam', action='store_true', default=True,
                        help='Use full Path-SAM2 + CTransPath model for inference (default: True). '
                             'Use --no_pathsam to convert to standard SAM2.')
    parser.add_argument('--no_pathsam', action='store_false', dest='use_pathsam',
                        help='Convert Path-SAM2 checkpoint to standard SAM2 (loses CTransPath features)')
    parser.add_argument('--ctranspath_checkpoint', type=str, 
                        default='models/ctranspath/ctranspath.pth',
                        help='Path to CTransPath pretrained weights (relative to project root)')
    parser.add_argument('--fusion_type', type=str, default='concat',
                        choices=['concat', 'attention'],
                        help='Fusion type used in Path-SAM2 training (concat or attention)')
    
    args = parser.parse_args()
    evaluate_segmentation(args)
