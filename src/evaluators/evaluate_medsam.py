"""
MedSAM Evaluation Script for BCSS Dataset

MedSAM (Nature Communications 2024) is trained on 1.57M medical image-mask pairs
across 10 imaging modalities. This script evaluates MedSAM on the BCSS 
histopathology segmentation task.

Key differences from SAM2:
- MedSAM uses original SAM ViT-B architecture (not SAM2 Hiera)
- MedSAM expects 1024x1024 input (resizes internally)
- MedSAM uses box prompts (point prompts less effective)

Usage:
    python src/evaluate_medsam.py --checkpoint models/medsam_checkpoints/medsam_vit_b.pth
    
    # With TTA:
    python src/evaluate_medsam.py --checkpoint models/medsam_checkpoints/medsam_vit_b.pth --use_tta
"""

import sys
import os

# Add project root to path for local imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
# Add MedSAM to path
medsam_root = os.path.join(project_root, 'MedSAM')
sys.path.insert(0, medsam_root)

import time
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from skimage import transform

from dataset import BCSSDataset
from sam_segmentation import calculate_metrics, get_prompts_from_mask
from device_utils import get_device


def load_medsam_model(checkpoint_path: str, device: str):
    """
    Load MedSAM model from checkpoint.
    
    MedSAM uses the original SAM ViT-B architecture from segment-anything.
    """
    # Import from MedSAM's segment_anything (not the official one)
    from segment_anything import sam_model_registry
    
    print(f"[MedSAM] Loading model from: {checkpoint_path}")
    medsam_model = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
    medsam_model = medsam_model.to(device)
    medsam_model.eval()
    print(f"[MedSAM] Model loaded successfully on {device}")
    
    return medsam_model


@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024, H, W, device):
    """
    Run MedSAM inference with box prompt.
    
    Args:
        medsam_model: Loaded MedSAM model
        img_embed: Image embedding from encoder (B, 256, 64, 64)
        box_1024: Bounding box in 1024x1024 scale [x1, y1, x2, y2]
        H, W: Original image dimensions for resizing output
        device: Device to use
    
    Returns:
        Binary segmentation mask at original resolution
    """
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (B, 1, 4)
    
    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,  # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
        multimask_output=False,
    )
    
    low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)
    
    low_res_pred = F.interpolate(
        low_res_pred,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, H, W)
    
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (H, W)
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    
    return medsam_seg, low_res_pred


def preprocess_image_for_medsam(image_np):
    """
    Preprocess image for MedSAM inference.
    
    MedSAM expects:
    - 1024x1024 resolution
    - Normalized to [0, 1]
    - Shape (3, H, W) tensor
    
    Args:
        image_np: Input image as numpy array (H, W, 3)
    
    Returns:
        img_1024_tensor: Preprocessed tensor
        H, W: Original dimensions
    """
    H, W = image_np.shape[:2]
    
    # Ensure 3 channels
    if len(image_np.shape) == 2:
        img_3c = np.repeat(image_np[:, :, None], 3, axis=-1)
    else:
        img_3c = image_np
    
    # Resize to 1024x1024
    img_1024 = transform.resize(
        img_3c, (1024, 1024), order=3,
        preserve_range=True, anti_aliasing=True
    ).astype(np.uint8)
    
    # Normalize to [0, 1]
    img_1024 = (img_1024 - img_1024.min()) / np.clip(
        img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
    )
    
    return img_1024, img_3c, H, W


def convert_box_format(box):
    """
    Convert box from [[x1, y1], [x2, y2]] to [x1, y1, x2, y2] format.
    
    The get_prompts_from_mask function returns box as shape (2, 2),
    but MedSAM expects [x1, y1, x2, y2].
    """
    box_np = np.array(box)
    if box_np.shape == (2, 2):
        # Convert from [[x1, y1], [x2, y2]] to [x1, y1, x2, y2]
        return [box_np[0, 0], box_np[0, 1], box_np[1, 0], box_np[1, 1]]
    elif box_np.shape == (4,):
        return box_np.tolist()
    else:
        raise ValueError(f"Unexpected box shape: {box_np.shape}")


def scale_box_to_1024(box, H, W):
    """
    Scale bounding box from original image size to 1024x1024.
    
    Args:
        box: [x1, y1, x2, y2] in original image coordinates
        H, W: Original image dimensions
    
    Returns:
        box_1024: [x1, y1, x2, y2] in 1024x1024 coordinates
    """
    box_flat = convert_box_format(box)
    box_np = np.array(box_flat)
    scale = np.array([1024/W, 1024/H, 1024/W, 1024/H])
    box_1024 = box_np * scale
    return box_1024


def predict_with_medsam(medsam_model, image_np, box, device):
    """
    Run full MedSAM prediction pipeline.
    
    Args:
        medsam_model: Loaded MedSAM model
        image_np: Input image (H, W, 3)
        box: Bounding box [x1, y1, x2, y2]
        device: Device to use
    
    Returns:
        Binary mask at original resolution
    """
    # Preprocess image
    img_1024, img_3c, H, W = preprocess_image_for_medsam(image_np)
    
    # Convert to tensor
    img_1024_tensor = (
        torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
    )
    
    # Scale box to 1024x1024
    box_1024 = scale_box_to_1024(box, H, W)
    box_1024 = np.array([box_1024])  # Add batch dimension
    
    # Get image embedding
    with torch.no_grad():
        image_embedding = medsam_model.image_encoder(img_1024_tensor)
    
    # Run inference
    medsam_seg, logits = medsam_inference(
        medsam_model, image_embedding, box_1024, H, W, device
    )
    
    return medsam_seg, logits


def predict_with_medsam_tta(medsam_model, image_np, box, device):
    """
    Run MedSAM prediction with test-time augmentation.
    
    Augmentations:
    - Original
    - Horizontal flip
    - Vertical flip
    - 90° rotation
    
    Args:
        medsam_model: Loaded MedSAM model
        image_np: Input image (H, W, 3)
        box: Bounding box [[x1, y1], [x2, y2]] or [x1, y1, x2, y2]
        device: Device to use
    
    Returns:
        Averaged binary mask at original resolution
    """
    H, W = image_np.shape[:2]
    all_logits = []
    
    # Convert box to flat format [x1, y1, x2, y2]
    box_flat = convert_box_format(box)
    x1, y1, x2, y2 = box_flat
    
    # Define augmentations and their inverse transforms
    # Each tuple: (name, image_aug_fn, logits_inv_fn, box_transform_fn)
    augmentations = [
        ('original', lambda x: x, lambda x: x, 
         lambda b: b),  # No change
        ('hflip', lambda x: np.fliplr(x).copy(), lambda x: np.fliplr(x), 
         lambda b: [W - b[2], b[1], W - b[0], b[3]]),  # Flip box x coords
        ('vflip', lambda x: np.flipud(x).copy(), lambda x: np.flipud(x),
         lambda b: [b[0], H - b[3], b[2], H - b[1]]),  # Flip box y coords
        ('rot90', lambda x: np.rot90(x, k=1).copy(), lambda x: np.rot90(x, k=-1),
         lambda b: [b[1], W - b[2], b[3], W - b[0]]),  # Rotate box coords
    ]
    
    for name, aug_fn, inv_fn, box_fn in augmentations:
        # Apply augmentation to image
        aug_image = aug_fn(image_np)
        # Transform box coordinates
        aug_box = box_fn(box_flat)
        
        # Get prediction
        _, logits = predict_with_medsam(medsam_model, aug_image, aug_box, device)
        
        # Inverse transform the logits back to original orientation
        inv_logits = inv_fn(logits)
        all_logits.append(inv_logits)
    
    # Average logits across augmentations
    avg_logits = np.mean(all_logits, axis=0)
    final_mask = (avg_logits > 0.5).astype(np.uint8)
    
    return final_mask, avg_logits


def evaluate_medsam(args):
    """Main evaluation function for MedSAM on BCSS dataset."""
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Resolve paths
    image_dir = os.path.join(project_root, args.image_dir)
    mask_dir = os.path.join(project_root, args.mask_dir)
    checkpoint_path = os.path.join(project_root, args.checkpoint)
    
    # Setup output directory
    output_dir = None
    if getattr(args, 'output_dir', None) and not getattr(args, 'print_only', False):
        output_dir = os.path.join(project_root, args.output_dir)
        os.makedirs(output_dir, exist_ok=True)
    
    # Verify checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: MedSAM checkpoint not found at: {checkpoint_path}")
        print("\nTo download MedSAM, run:")
        print("  bash scripts/download_medsam.sh")
        sys.exit(1)
    
    # Load dataset
    bcss_dataset = BCSSDataset(image_dir=image_dir, mask_dir=mask_dir, split=args.split)
    print(f"Loaded {len(bcss_dataset)} samples from {args.split} split")
    
    # Setup device
    device = get_device(force_cpu=args.force_cpu) if hasattr(args, 'force_cpu') else get_device()
    print(f"Using device: {device}")
    
    # Load MedSAM model
    medsam_model = load_medsam_model(checkpoint_path, device)
    
    # Evaluation settings
    num_samples = len(bcss_dataset) if args.max_samples is None else min(args.max_samples, len(bcss_dataset))
    class_names = bcss_dataset.class_names
    class_dice_scores = {cid: [] for cid in class_names if cid != 0}
    class_iou_scores = {cid: [] for cid in class_names if cid != 0}  # Added IoU tracking
    sample_results = []
    
    print(f"\n{'='*60}")
    print(f"MedSAM Evaluation")
    print(f"{'='*60}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: BCSS {args.split} split ({num_samples} samples)")
    print(f"TTA: {'Enabled' if args.use_tta else 'Disabled'}")
    print(f"{'='*60}\n")
    
    # Evaluation loop
    loop_start = time.time()
    
    for i in tqdm(range(num_samples), desc="Evaluating", disable=not args.tqdm):
        sample = bcss_dataset[i]
        image = sample['image_np']
        gt_mask = sample['mask'].numpy()
        unique_classes = np.unique(gt_mask)
        
        # Preprocess image ONCE per sample (expensive operation)
        img_1024, img_3c, H, W = preprocess_image_for_medsam(image)
        img_1024_tensor = (
            torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
        )
        
        # Encode image ONCE per sample
        with torch.no_grad():
            image_embedding = medsam_model.image_encoder(img_1024_tensor)
        
        for class_id in unique_classes:
            if class_id == 0:
                continue
            
            class_name = class_names.get(class_id)
            if not class_name:
                continue
            
            # Get binary ground truth mask
            binary_gt_mask = (gt_mask == class_id).astype(np.uint8)
            
            # Get box prompt from ground truth
            prompts = get_prompts_from_mask(binary_gt_mask)
            if 'box' not in prompts:
                continue
            
            box = prompts['box']  # [[x1, y1], [x2, y2]]
            
            # Run prediction (use pre-computed embedding)
            if args.use_tta:
                predicted_mask, _ = predict_with_medsam_tta(
                    medsam_model, image, box, device
                )
            else:
                # Use pre-computed embedding for non-TTA case
                box_1024 = scale_box_to_1024(box, H, W)
                box_1024 = np.array([box_1024])  # Add batch dimension
                predicted_mask, _ = medsam_inference(
                    medsam_model, image_embedding, box_1024, H, W, device
                )
            
            # Calculate metrics
            dice, iou = calculate_metrics(predicted_mask, binary_gt_mask)
            class_dice_scores[class_id].append(dice)
            class_iou_scores[class_id].append(iou)  # Track IoU
            sample_results.append({
                'sample_idx': int(i),
                'class_id': int(class_id),
                'class_name': class_name,
                'dice': float(dice),
                'iou': float(iou),
                'image_path': bcss_dataset.image_files[i]
            })
    
    elapsed = time.time() - loop_start
    print(f"\nEvaluation completed in {elapsed:.1f}s ({elapsed/num_samples:.2f}s per sample)")
    
    # Calculate and report results
    results = {}
    for cid, cname in class_names.items():
        if cid == 0:
            continue
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
    
    # Add metadata
    results['_metadata'] = {
        'model': 'MedSAM',
        'checkpoint': args.checkpoint,
        'split': args.split,
        'num_samples': num_samples,
        'tta': args.use_tta,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    
    # Print results
    print("\n" + "="*60)
    print("Results: Per-Class Metrics (Dice | IoU)")
    print("="*60)
    for cname, data in results.items():
        if cname.startswith('_'):
            continue
        if cname == 'overall':
            print("-"*50)
            print(f"{'OVERALL':15s}: Dice={data['dice']:.4f} | IoU={data['iou']:.4f}")
        else:
            print(f"{cname:15s}: {data['dice']:.4f} ± {data['dice_std']:.4f} | {data['iou']:.4f} ± {data['iou_std']:.4f} (n={data['count']})")
    print("="*60)
    
    # Save results
    if output_dir is not None:
        metrics_path = os.path.join(output_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nMetrics saved to: {metrics_path}")
        
        # Save per-sample results
        samples_path = os.path.join(output_dir, 'sample_results.json')
        with open(samples_path, 'w') as f:
            json.dump(sample_results, f, indent=2)
        print(f"Sample results saved to: {samples_path}")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Evaluate MedSAM on BCSS histopathology dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  python src/evaluate_medsam.py --checkpoint models/medsam_checkpoints/medsam_vit_b.pth

  # With TTA and save results
  python src/evaluate_medsam.py \\
      --checkpoint models/medsam_checkpoints/medsam_vit_b.pth \\
      --use_tta \\
      --output_dir results/medsam_tta_eval
      
  # Quick test (2 samples)
  python src/evaluate_medsam.py \\
      --checkpoint models/medsam_checkpoints/medsam_vit_b.pth \\
      --max_samples 2
        """
    )
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to MedSAM checkpoint (relative to project root)')
    parser.add_argument('--image_dir', type=str, default='data/bcss/images',
                        help='Path to image directory (relative to project root)')
    parser.add_argument('--mask_dir', type=str, default='data/bcss/masks',
                        help='Path to mask directory (relative to project root)')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate on')
    parser.add_argument('--use_tta', action='store_true',
                        help='Enable test-time augmentation (hflip, vflip, rot90)')
    parser.add_argument('--force_cpu', action='store_true',
                        help='Force CPU usage even if GPU is available')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Limit number of samples (for debugging)')
    parser.add_argument('--tqdm', action='store_true',
                        help='Enable tqdm progress bar')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save results (relative to project root)')
    parser.add_argument('--print_only', action='store_true',
                        help='Print results without saving files')
    
    args = parser.parse_args()
    evaluate_medsam(args)
