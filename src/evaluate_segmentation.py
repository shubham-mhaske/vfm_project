import sys
import os
# Add project root to path for local imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
# Add sam2 repo root to path BEFORE project root for training module access
sam2_root = os.path.join(project_root, 'sam2')
sys.path.insert(0, sam2_root)

import torch
import numpy as np
from tqdm import tqdm
from dataset import BCSSDataset
from sam_segmentation import get_sam2_predictor, get_predicted_mask, calculate_metrics
import argparse
from device_utils import get_device
from training.utils.train_utils import register_omegaconf_resolvers

def evaluate_segmentation(args):
    """Runs the SAM 2 segmentation and evaluation on the full test set (or a subset)."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    image_dir = os.path.join(project_root, args.image_dir)
    mask_dir = os.path.join(project_root, args.mask_dir)

    if args.verbose:
        print("[INIT] Building BCSSDataset...", flush=True)
    bcss_dataset = BCSSDataset(image_dir=image_dir, mask_dir=mask_dir, split=args.split)
    if args.verbose:
        print(f"[INIT] Dataset split '{args.split}' size: {len(bcss_dataset)}", flush=True)

    device = get_device(force_cpu=args.force_cpu) if hasattr(args, 'force_cpu') else get_device()
    print(f"Using device: {device}", flush=True)

    if args.verbose:
        print("[INIT] Registering OmegaConf resolvers...", flush=True)
    register_omegaconf_resolvers()

    model_cfg = os.path.join(project_root, args.model_cfg)
    checkpoint_path = os.path.join(project_root, args.checkpoint)
    if args.verbose:
        print(f"[LOAD] Loading checkpoint from {checkpoint_path} ...", flush=True)
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if args.verbose:
        print("[LOAD] Checkpoint loaded in memory.", flush=True)

    if 'model' in ckpt and isinstance(ckpt['model'], dict):
        print(f"Loading finetuned checkpoint from epoch {ckpt.get('epoch', 'unknown')}", flush=True)
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        import sam2
        if args.verbose:
            print("[BUILD] Building SAM2 model (no weights)...", flush=True)
        sam2_pkg_root = sam2.__path__[0]
        relative_cfg_path = os.path.relpath(model_cfg, sam2_pkg_root).replace('\\', '/')
        model = build_sam2(relative_cfg_path, ckpt_path=None, device=device)
        if args.verbose:
            print("[LOAD] Loading finetuned state dict into model...", flush=True)
        missing_keys, unexpected_keys = model.load_state_dict(ckpt['model'], strict=False)
        if missing_keys:
            print(f"Missing keys: {missing_keys}", flush=True)
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}", flush=True)
        predictor = SAM2ImagePredictor(model)
    else:
        if args.verbose:
            print("[BUILD] Using standard predictor loader...", flush=True)
        predictor = get_sam2_predictor(model_cfg, checkpoint_path, device)
    if args.verbose:
        print("[READY] Predictor initialized.", flush=True)

    num_samples_total = len(bcss_dataset)
    num_samples = num_samples_total if args.max_samples is None else min(args.max_samples, num_samples_total)
    if args.max_samples is not None:
        print(f"[CONFIG] Limiting evaluation to {num_samples} / {num_samples_total} samples", flush=True)

    print(f"Prompt type: {args.prompt_type}", flush=True)
    print(f"Use negative points: {args.use_neg_points}", flush=True)
    print(f"Evaluating on {num_samples} samples...", flush=True)

    total_dice = 0.0
    total_iou = 0.0

    for i in tqdm(range(num_samples), disable=not args.tqdm):
        sample = bcss_dataset[i]
        image = sample['image_np']
        gt_mask = sample['mask'].numpy()

        predicted_mask, binary_mask, _, _ = get_predicted_mask(
            predictor, image, gt_mask,
            prompt_type=args.prompt_type,
            use_neg_points=args.use_neg_points
        )

        dice, iou = calculate_metrics(predicted_mask, binary_mask)
        total_dice += dice
        total_iou += iou

        if args.verbose and (i + 1) % 10 == 0:
            print(f"[PROGRESS] Processed {i+1}/{num_samples}", flush=True)

    avg_dice = total_dice / num_samples
    avg_iou = total_iou / num_samples
    print(f"Average Dice Score: {avg_dice:.4f}", flush=True)
    print(f"Average IoU: {avg_iou:.4f}", flush=True)

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
    args = parser.parse_args()
    evaluate_segmentation(args)
