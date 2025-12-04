#!/usr/bin/env python3
"""
Generate Qualitative Results with Best-Performing Samples

This script:
1. Runs evaluation on all test samples and saves per-image metrics
2. Selects the best-performing samples per class
3. Generates publication-quality qualitative comparison figures

Usage:
    python scripts/analysis/generate_best_qualitative.py [--quick]
    
For HPRC:
    sbatch scripts/slurm/generate_best_qualitative.slurm
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from pathlib import Path
from tqdm import tqdm
import torch
import cv2

# Force unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'

def log(msg):
    print(msg, flush=True)

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
sys.path.insert(0, str(PROJECT_ROOT / 'sam2'))

from dataset import BCSSDataset
from sam_segmentation import (
    get_sam2_predictor,
    get_prompts_from_mask,
    get_predicted_mask_from_prompts,
    postprocess_mask,
    calculate_metrics
)
from device_utils import get_device

# Output paths
OUTPUT_DIR = PROJECT_ROOT / 'results' / 'best_qualitative'
FIGURES_DIR = PROJECT_ROOT / 'results' / 'figures' / 'presentation'

# Model configurations - only SAM2 variants for speed
MODELS = {
    'sam2_centroid': {
        'name': 'SAM2 (Centroid)',
        'prompt_type': 'centroid',
        'use_neg_points': False,
    },
    'sam2_box': {
        'name': 'SAM2 (Box)',
        'prompt_type': 'box',
        'use_neg_points': False,
    },
    'sam2_box_neg': {
        'name': 'SAM2 (Box+Neg)',
        'prompt_type': 'box',
        'use_neg_points': True,
    },
}

# Class colors for visualization
CLASS_COLORS = {
    0: [0, 0, 0],        # background - black
    1: [255, 0, 0],      # tumor - red
    2: [0, 255, 0],      # stroma - green
    3: [0, 0, 255],      # lymphocyte - blue
    4: [255, 255, 0],    # necrosis - yellow
    5: [255, 0, 255],    # blood_vessel - magenta
}

CLASS_NAMES = {
    1: 'tumor',
    2: 'stroma', 
    3: 'lymphocyte',
    4: 'necrosis',
    5: 'blood_vessel'
}

# Map internal class IDs to display IDs (BCSS uses 18 for blood vessel)
CLASS_ID_MAP = {1: 1, 2: 2, 3: 3, 4: 4, 18: 5}
DISPLAY_TO_INTERNAL = {v: k for k, v in CLASS_ID_MAP.items()}


def create_colored_mask(mask, alpha=0.6):
    """Convert class mask to RGB colored overlay."""
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id, color in CLASS_COLORS.items():
        colored[mask == class_id] = color
    
    return colored


def create_overlay(image, mask, color, alpha=0.5):
    """Create an overlay of a colored mask on an image."""
    overlay = image.copy().astype(np.float32)
    mask_bool = mask.astype(bool)
    overlay[mask_bool] = (1 - alpha) * image[mask_bool] + alpha * np.array(color)
    return overlay.astype(np.uint8)


def load_sam2_predictor(device, sam2_dir_override=None):
    """Load SAM2 predictor with proper directory handling."""
    # SAM2 needs to be loaded from its directory for hydra configs
    original_dir = os.getcwd()
    
    # Use override if provided, otherwise default
    if sam2_dir_override:
        sam2_dir = sam2_dir_override
    else:
        sam2_dir = str(PROJECT_ROOT / 'sam2')
    
    log(f"  Changing to SAM2 directory: {sam2_dir}")
    os.chdir(sam2_dir)
    
    try:
        # Register OmegaConf resolvers
        try:
            from training.utils.train_utils import register_omegaconf_resolvers
            register_omegaconf_resolvers()
        except:
            pass
        
        # Load with relative config path
        sam_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        sam_ckpt = os.path.join(sam2_dir, 'checkpoints', 'sam2.1_hiera_large.pt')
        
        predictor = get_sam2_predictor(sam_cfg, sam_ckpt, device)
        log("  ✓ SAM2 loaded successfully")
        
    finally:
        os.chdir(original_dir)
    
    return predictor


def run_per_image_evaluation(predictor, model_config, dataset, device, max_samples=None):
    """Run evaluation and return per-image metrics."""
    log(f"\n  Evaluating: {model_config['name']}")
    
    results = []
    n_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    log(f"    Processing {n_samples} samples...")
    
    for idx in tqdm(range(n_samples), desc=f"    {model_config['name']}"):
        if idx % 10 == 0:
            log(f"    [Progress] Sample {idx+1}/{n_samples}")
        sample = dataset[idx]
        
        # Handle both tensor and numpy outputs
        image = sample['image_np']
        if hasattr(image, 'numpy'):
            image = image.numpy()
        
        gt_mask = sample['mask']
        if hasattr(gt_mask, 'numpy'):
            gt_mask = gt_mask.numpy()
        
        filename = sample['filename']
        
        unique_classes = sample['unique_classes']
        if hasattr(unique_classes, 'numpy'):
            unique_classes = unique_classes.numpy()
        unique_classes = unique_classes[unique_classes != 0]
        
        sample_results = {
            'filename': filename,
            'idx': idx,
            'per_class': {},
            'image': image,  # Store for visualization
            'gt_mask': gt_mask,
        }
        
        for class_id in unique_classes:
            # Map internal class ID to display ID
            display_id = CLASS_ID_MAP.get(int(class_id), None)
            if display_id is None or display_id not in CLASS_NAMES:
                continue
                
            class_name = CLASS_NAMES[display_id]
            binary_gt_mask = (gt_mask == class_id).astype(np.uint8)
            
            if binary_gt_mask.sum() < 100:  # Skip very small regions
                continue
            
            # Get prompts from GT mask
            prompts = get_prompts_from_mask(binary_gt_mask)
            if not prompts or model_config['prompt_type'] not in prompts:
                continue
            
            # Run SAM prediction
            try:
                predicted_mask, _, _ = get_predicted_mask_from_prompts(
                    predictor,
                    image,
                    prompts,
                    prompt_type=model_config['prompt_type'],
                    use_neg_points=model_config['use_neg_points']
                )
                final_mask = postprocess_mask(predicted_mask)
                
                # Calculate metrics
                dice, iou = calculate_metrics(final_mask, binary_gt_mask)
                
                sample_results['per_class'][class_name] = {
                    'class_id': int(class_id),
                    'display_id': int(display_id),
                    'dice': float(dice),
                    'iou': float(iou),
                    'predicted_mask': final_mask,
                    'gt_mask': binary_gt_mask,
                    'prompts': prompts,
                }
            except Exception as e:
                log(f"    Warning: Failed for {filename}/{class_name}: {e}")
                continue
        
        if sample_results['per_class']:
            # Calculate overall sample dice (average across classes)
            dices = [v['dice'] for v in sample_results['per_class'].values()]
            sample_results['avg_dice'] = float(np.mean(dices))
            results.append(sample_results)
    
    log(f"    Completed: {len(results)} samples with valid results")
    return results


def select_best_samples(all_results, n_per_class=3, n_overall=5):
    """Select best-performing samples per class and overall."""
    
    best_samples = {
        'overall': [],
        'per_class': {name: [] for name in CLASS_NAMES.values()}
    }
    
    # Best overall samples (by average dice)
    sorted_overall = sorted(all_results, key=lambda x: x['avg_dice'], reverse=True)
    best_samples['overall'] = sorted_overall[:n_overall]
    
    # Best per-class samples
    for class_name in CLASS_NAMES.values():
        class_results = []
        for sample in all_results:
            if class_name in sample['per_class']:
                class_results.append({
                    'sample': sample,
                    'dice': sample['per_class'][class_name]['dice']
                })
        
        sorted_class = sorted(class_results, key=lambda x: x['dice'], reverse=True)
        best_samples['per_class'][class_name] = [x['sample'] for x in sorted_class[:n_per_class]]
    
    return best_samples


def create_method_comparison_figure(best_samples, all_model_results, output_path):
    """Create a method comparison figure with best samples."""
    log("\n  Creating method comparison figure...")
    
    # Get top 4 overall samples
    samples = best_samples['overall'][:4]
    n_samples = len(samples)
    model_keys = list(all_model_results.keys())
    n_methods = len(model_keys)
    
    # Figure: rows = samples, cols = [image, gt, method1, method2, method3]
    fig, axes = plt.subplots(n_samples, n_methods + 2, figsize=(3.5 * (n_methods + 2), 3.5 * n_samples))
    
    for row, sample in enumerate(samples):
        filename = sample['filename']
        image = sample['image']
        gt_mask = sample['gt_mask']
        
        # Column 0: Original image
        axes[row, 0].imshow(image)
        if row == 0:
            axes[row, 0].set_title('Original', fontsize=14, fontweight='bold')
        axes[row, 0].axis('off')
        
        # Column 1: Ground truth
        gt_colored = create_colored_mask(gt_mask)
        axes[row, 1].imshow(image)
        axes[row, 1].imshow(gt_colored, alpha=0.5)
        if row == 0:
            axes[row, 1].set_title('Ground Truth', fontsize=14, fontweight='bold')
        dice_str = f"Avg: {sample['avg_dice']:.3f}"
        axes[row, 1].text(0.5, -0.08, dice_str, transform=axes[row, 1].transAxes,
                         ha='center', fontsize=10, color='black')
        axes[row, 1].axis('off')
        
        # Columns 2+: Model predictions
        for col, model_key in enumerate(model_keys):
            ax = axes[row, col + 2]
            model_results = all_model_results[model_key]
            
            # Find this sample in model results
            model_sample = None
            for s in model_results:
                if s['filename'] == filename:
                    model_sample = s
                    break
            
            if model_sample:
                # Reconstruct full prediction mask
                pred_mask = np.zeros_like(gt_mask)
                sample_dices = []
                for class_name, class_data in model_sample['per_class'].items():
                    class_id = class_data['class_id']
                    pred_mask[class_data['predicted_mask'] > 0] = class_id
                    sample_dices.append(class_data['dice'])
                
                pred_colored = create_colored_mask(pred_mask)
                ax.imshow(image)
                ax.imshow(pred_colored, alpha=0.5)
                
                if sample_dices:
                    mean_dice = np.mean(sample_dices)
                    color = 'green' if mean_dice > 0.5 else 'orange' if mean_dice > 0.3 else 'red'
                    ax.text(0.5, -0.08, f"Dice: {mean_dice:.3f}", transform=ax.transAxes,
                           ha='center', fontsize=11, color=color, fontweight='bold')
            else:
                ax.imshow(image)
                ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes, ha='center', va='center')
            
            if row == 0:
                ax.set_title(MODELS[model_key]['name'], fontsize=14, fontweight='bold')
            ax.axis('off')
    
    plt.suptitle('Best-Performing Samples: Method Comparison', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(str(output_path).replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    log(f"    ✓ Saved: {output_path}")


def create_per_class_figure(best_samples, model_key, model_results, output_path):
    """Create per-class best samples figure."""
    log("\n  Creating per-class figure...")
    
    class_names_list = list(CLASS_NAMES.values())
    n_classes = len(class_names_list)
    n_samples = 2  # Show 2 best samples per class
    
    fig, axes = plt.subplots(n_classes, n_samples * 3, figsize=(4 * n_samples * 3, 3.5 * n_classes))
    
    for row, class_name in enumerate(class_names_list):
        samples = best_samples['per_class'][class_name][:n_samples]
        
        for sample_idx, sample in enumerate(samples):
            col_base = sample_idx * 3
            filename = sample['filename']
            image = sample['image']
            gt_mask = sample['gt_mask']
            
            # Find in model results
            model_sample = None
            for s in model_results:
                if s['filename'] == filename:
                    model_sample = s
                    break
            
            # Get display class ID
            display_id = list(CLASS_NAMES.keys())[list(CLASS_NAMES.values()).index(class_name)]
            internal_id = DISPLAY_TO_INTERNAL.get(display_id, display_id)
            
            # Original
            axes[row, col_base].imshow(image)
            if row == 0:
                axes[row, col_base].set_title(f'Sample {sample_idx+1}\nOriginal', fontsize=12)
            axes[row, col_base].axis('off')
            
            # Ground Truth (class-specific)
            if class_name in sample['per_class']:
                class_data = sample['per_class'][class_name]
                gt_binary = class_data['gt_mask']
                gt_overlay = create_overlay(image, gt_binary, CLASS_COLORS[display_id], alpha=0.6)
                axes[row, col_base + 1].imshow(gt_overlay)
            else:
                axes[row, col_base + 1].imshow(image)
            
            if row == 0:
                axes[row, col_base + 1].set_title('Ground Truth', fontsize=12)
            axes[row, col_base + 1].axis('off')
            
            # Prediction
            if model_sample and class_name in model_sample['per_class']:
                class_data = model_sample['per_class'][class_name]
                pred_overlay = create_overlay(image, class_data['predicted_mask'], 
                                             [30, 144, 255], alpha=0.6)
                axes[row, col_base + 2].imshow(pred_overlay)
                
                # Draw box prompt if available
                if 'prompts' in class_data and 'box' in class_data['prompts']:
                    box = class_data['prompts']['box']
                    rect = plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                                        linewidth=2, edgecolor='lime', facecolor='none')
                    axes[row, col_base + 2].add_patch(rect)
                
                dice = class_data['dice']
                color = 'green' if dice > 0.6 else 'orange' if dice > 0.4 else 'red'
                axes[row, col_base + 2].text(0.5, -0.08, f"Dice: {dice:.3f}", 
                                             transform=axes[row, col_base + 2].transAxes,
                                             ha='center', fontsize=11, color=color, fontweight='bold')
            else:
                axes[row, col_base + 2].imshow(image)
                axes[row, col_base + 2].text(0.5, 0.5, 'N/A', transform=axes[row, col_base + 2].transAxes,
                                             ha='center', va='center')
            
            if row == 0:
                axes[row, col_base + 2].set_title('Prediction', fontsize=12)
            axes[row, col_base + 2].axis('off')
        
        # Row label
        axes[row, 0].set_ylabel(class_name.upper(), fontsize=14, fontweight='bold', rotation=0, labelpad=60)
    
    plt.suptitle(f'Best Per-Class Results: {MODELS[model_key]["name"]}', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(str(output_path).replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    log(f"    ✓ Saved: {output_path}")


def create_success_failure_figure(all_results, model_key, output_path):
    """Create success vs failure comparison."""
    log("\n  Creating success/failure figure...")
    
    # Sort by dice
    sorted_results = sorted(all_results, key=lambda x: x['avg_dice'], reverse=True)
    
    # Get top 3 and bottom 3
    successes = sorted_results[:3]
    failures = sorted_results[-3:]
    
    fig, axes = plt.subplots(2, 6, figsize=(18, 7))
    
    for row_idx, (samples, label, color) in enumerate([
        (successes, 'SUCCESS', 'green'),
        (failures, 'FAILURE', 'red')
    ]):
        for col, sample in enumerate(samples):
            base_col = col * 2
            
            image = sample['image']
            gt_mask = sample['gt_mask']
            
            # Original + GT overlay
            ax1 = axes[row_idx, base_col]
            gt_colored = create_colored_mask(gt_mask)
            ax1.imshow(image)
            ax1.imshow(gt_colored, alpha=0.4)
            ax1.set_title(f"{label} {col+1}: GT", fontsize=11)
            ax1.axis('off')
            
            # Prediction overlay
            ax2 = axes[row_idx, base_col + 1]
            pred_mask = np.zeros_like(gt_mask)
            for class_name, class_data in sample['per_class'].items():
                pred_mask[class_data['predicted_mask'] > 0] = class_data['class_id']
            pred_colored = create_colored_mask(pred_mask)
            
            ax2.imshow(image)
            ax2.imshow(pred_colored, alpha=0.4)
            dice = sample['avg_dice']
            ax2.set_title(f"Dice: {dice:.3f}", fontsize=11, color=color, fontweight='bold')
            ax2.axis('off')
    
    # Row labels
    fig.text(0.02, 0.75, 'HIGH DICE\n(Success)', fontsize=14, fontweight='bold', color='green',
             ha='center', va='center', rotation=90)
    fig.text(0.02, 0.25, 'LOW DICE\n(Failure)', fontsize=14, fontweight='bold', color='red',
             ha='center', va='center', rotation=90)
    
    plt.suptitle(f'Success vs Failure Cases: {MODELS[model_key]["name"]}', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0.03, 0, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(str(output_path).replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    log(f"    ✓ Saved: {output_path}")


def save_metrics_summary(all_model_results, output_path):
    """Save per-image metrics for all models."""
    summary = {}
    
    for model_key, results in all_model_results.items():
        model_summary = {
            'model_name': MODELS[model_key]['name'],
            'per_image': []
        }
        
        for sample in results:
            img_summary = {
                'filename': sample['filename'],
                'avg_dice': sample['avg_dice'],
                'per_class': {}
            }
            for class_name, class_data in sample['per_class'].items():
                img_summary['per_class'][class_name] = {
                    'dice': class_data['dice'],
                    'iou': class_data['iou']
                }
            model_summary['per_image'].append(img_summary)
        
        # Sort by dice for easy reference
        model_summary['per_image'].sort(key=lambda x: x['avg_dice'], reverse=True)
        
        # Add aggregate stats
        all_dices = [s['avg_dice'] for s in results]
        if all_dices:
            model_summary['aggregate'] = {
                'mean_dice': float(np.mean(all_dices)),
                'std_dice': float(np.std(all_dices)),
                'min_dice': float(np.min(all_dices)),
                'max_dice': float(np.max(all_dices)),
                'n_samples': len(results)
            }
        
        summary[model_key] = model_summary
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    log(f"  Saved metrics summary: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate best-performing qualitative results")
    parser.add_argument('--quick', action='store_true', 
                        help="Run on subset of samples for quick testing")
    parser.add_argument('--max-samples', type=int, default=None,
                        help="Maximum number of samples to evaluate")
    parser.add_argument('--data-dir', type=str, default=None,
                        help="Path to data directory (default: PROJECT_ROOT/data/bcss)")
    parser.add_argument('--sam2-dir', type=str, default=None,
                        help="Path to SAM2 directory with checkpoints (default: PROJECT_ROOT/sam2)")
    args = parser.parse_args()
    
    # Store args globally for load_sam2_predictor
    global ARGS
    ARGS = args
    
    log("=" * 70)
    log("Generating Best-Performing Qualitative Results")
    log("=" * 70)
    
    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get device
    device = get_device()
    log(f"Using device: {device}")
    
    # Load dataset
    if args.data_dir:
        data_base = Path(args.data_dir)
    else:
        data_base = PROJECT_ROOT / 'data' / 'bcss'
    image_dir = data_base / 'images'
    mask_dir = data_base / 'masks'
    log(f"Loading data from: {data_base}")
    dataset = BCSSDataset(image_dir=str(image_dir), mask_dir=str(mask_dir), split='test')
    log(f"Loaded {len(dataset)} test samples")
    
    # Determine number of samples
    max_samples = args.max_samples
    if args.quick:
        max_samples = 10
        log(f"Quick mode: evaluating only {max_samples} samples")
    
    # Load SAM2 predictor once
    log("\nLoading SAM2 model...")
    predictor = load_sam2_predictor(device, args.sam2_dir)
    
    # Run evaluation for all model configurations
    log("\n" + "=" * 60)
    log("Running per-image evaluation...")
    log("=" * 60)
    
    all_model_results = {}
    for model_key, model_config in MODELS.items():
        results = run_per_image_evaluation(predictor, model_config, dataset, device, max_samples)
        all_model_results[model_key] = results
        log(f"    {model_config['name']}: {len(results)} samples evaluated")
    
    # Save per-image metrics
    save_metrics_summary(all_model_results, OUTPUT_DIR / 'per_image_metrics.json')
    
    # Select best samples (using sam2_box_neg as reference since it's best)
    log("\nSelecting best-performing samples...")
    best_samples = select_best_samples(all_model_results['sam2_box_neg'], n_per_class=3, n_overall=5)
    log(f"  Overall best samples: {len(best_samples['overall'])}")
    for class_name, samples in best_samples['per_class'].items():
        log(f"  Best {class_name}: {len(samples)} samples")
    
    # Generate figures
    log("\n" + "=" * 60)
    log("Generating Figures...")
    log("=" * 60)
    
    # 1. Method comparison with best samples
    log("\n  [1/3] Creating method comparison figure...")
    create_method_comparison_figure(
        best_samples,
        all_model_results,
        str(FIGURES_DIR / 'best_method_comparison.png')
    )
    log("    -> Saved: best_method_comparison.png")
    
    # 2. Per-class best samples
    log("\n  [2/3] Creating per-class figure...")
    create_per_class_figure(
        best_samples,
        'sam2_box_neg',
        all_model_results['sam2_box_neg'],
        str(FIGURES_DIR / 'best_per_class.png')
    )
    log("    -> Saved: best_per_class.png")
    
    # 3. Success vs Failure
    log("\n  [3/3] Creating success/failure figure...")
    create_success_failure_figure(
        all_model_results['sam2_box_neg'],
        'sam2_box_neg',
        str(FIGURES_DIR / 'best_success_failure.png')
    )
    log("    -> Saved: best_success_failure.png")
    
    log("\n" + "=" * 70)
    log("Done! Generated files:")
    log("=" * 70)
    log(f"  Metrics: {OUTPUT_DIR / 'per_image_metrics.json'}")
    log(f"  Figures:")
    for f in FIGURES_DIR.glob('best_*.png'):
        log(f"    - {f.name}")


if __name__ == '__main__':
    main()
