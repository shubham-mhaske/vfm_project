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

def evaluate_segmentation(args):
    """Runs the SAM 2 segmentation and evaluation on the full test set (or a subset)."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
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

    if 'model' in ckpt and isinstance(ckpt['model'], dict):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        model = build_sam2(args.model_cfg, ckpt_path=None, device=device)
        model.load_state_dict(ckpt['model'], strict=False)
        predictor = SAM2ImagePredictor(model)
    else:
        predictor = get_sam2_predictor(args.model_cfg, checkpoint_path, device)

    num_samples = len(bcss_dataset) if args.max_samples is None else min(args.max_samples, len(bcss_dataset))
    
    class_thresholds = {}
    if args.threshold_config and os.path.exists(args.threshold_config):
        with open(args.threshold_config, 'r') as f:
            class_thresholds = json.load(f)
        print(f"Loaded custom thresholds from: {args.threshold_config}")

    class_names = bcss_dataset.class_names
    class_dice_scores = {cid: [] for cid in class_names if cid != 0}
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
            sample_results.append({'sample_idx': i, 'class_id': class_id, 'dice': dice, 'image_path': bcss_dataset.image_files[i]})

    # --- Metrics Calculation and Reporting ---
    results = {}
    for cid, cname in class_names.items():
        if cid == 0: continue
        scores = class_dice_scores.get(cid, [])
        results[cname] = {'dice': np.mean(scores) if scores else 0, 'std': np.std(scores) if scores else 0, 'count': len(scores)}
    
    all_scores = [s for scores in class_dice_scores.values() for s in scores]
    results['overall'] = np.mean(all_scores) if all_scores else 0
    
    print("\n--- Per-Class Dice Scores ---")
    for cname, data in results.items():
        if cname != 'overall':
            print(f"{cname:15s}: {data['dice']:.4f} (count: {data['count']})")
    print(f"---------------------------------")
    print(f"Overall Avg Dice: {results['overall']:.4f}")

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
    args = parser.parse_args()
    evaluate_segmentation(args)
