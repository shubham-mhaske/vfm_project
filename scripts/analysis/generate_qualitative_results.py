#!/usr/bin/env python3
"""
Generate Qualitative Results: Side-by-side visualizations of test images with 
segmentation predictions from best models (SAM2 Box+Neg, MedSAM+TTA).

Creates publication-quality figures showing:
1. Multi-panel comparisons of segmentation methods
2. Per-class segmentation examples  
3. Success and failure cases
4. CLIP classification visualization with region crops

Usage:
    python scripts/analysis/generate_qualitative_results.py
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import cv2
from PIL import Image

# Add project paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, os.path.join(project_root, 'sam2'))

# Set matplotlib style for academic figures
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Color scheme for tissue classes
CLASS_COLORS = {
    0: [0, 0, 0],           # Background - black
    1: [255, 0, 0],         # Tumor - red
    2: [0, 255, 0],         # Stroma - green
    3: [0, 0, 255],         # Lymphocyte - blue
    4: [255, 255, 0],       # Necrosis - yellow
    18: [255, 0, 255],      # Blood Vessel - magenta
}

CLASS_NAMES = {
    0: 'Background',
    1: 'Tumor',
    2: 'Stroma', 
    3: 'Lymphocyte',
    4: 'Necrosis',
    18: 'Blood Vessel'
}

CLASS_NAME_TO_ID = {v.lower().replace(' ', '_'): k for k, v in CLASS_NAMES.items()}


def load_models():
    """Load SAM2 and MedSAM predictors."""
    from device_utils import get_device
    from sam_segmentation import get_sam2_predictor
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Need to change to sam2 directory for hydra config resolution
    original_dir = os.getcwd()
    sam2_dir = os.path.join(project_root, "sam2")
    os.chdir(sam2_dir)
    
    # Register OmegaConf resolvers if available
    try:
        from training.utils.train_utils import register_omegaconf_resolvers
        register_omegaconf_resolvers()
    except:
        pass
    
    # Load SAM2 - use relative config name for hydra
    sam2_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"  # Relative path for hydra
    sam2_ckpt = os.path.join(project_root, "sam2/checkpoints/sam2.1_hiera_large.pt")
    sam2_predictor = get_sam2_predictor(sam2_cfg, sam2_ckpt, device)
    print("✓ SAM2 loaded")
    
    # Change back to original directory
    os.chdir(original_dir)
    
    # Load MedSAM
    medsam_predictor = None
    try:
        medsam_ckpt = os.path.join(project_root, "models/medsam_checkpoints/medsam_vit_b.pth")
        if os.path.exists(medsam_ckpt):
            sys.path.insert(0, os.path.join(project_root, 'MedSAM'))
            from segment_anything import sam_model_registry
            medsam_model = sam_model_registry['vit_b'](checkpoint=medsam_ckpt)
            medsam_model = medsam_model.to(device)
            medsam_model.eval()
            from segment_anything import SamPredictor
            medsam_predictor = SamPredictor(medsam_model)
            print("✓ MedSAM loaded")
    except Exception as e:
        print(f"⚠ Could not load MedSAM: {e}")
    
    return sam2_predictor, medsam_predictor, device


def get_prompts_from_mask(binary_mask, num_neg_points=5, neg_point_margin=15):
    """Generate prompts (box, centroid, multi-point, negative points) from binary mask."""
    prompts = {}
    
    contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return prompts
    
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    prompts['box'] = np.array([x, y, x + w, y + h])
    
    # Centroid
    M = cv2.moments(largest_contour)
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        prompts['centroid'] = np.array([[cx, cy]])
    
    # Multi-point (boundary points)
    if len(largest_contour) >= 5:
        step = max(1, len(largest_contour) // 5)
        points = largest_contour[::step, 0, :][:5]
        prompts['multi_point'] = points
    
    # Negative points (outside bounding box)
    h_img, w_img = binary_mask.shape
    neg_points = []
    np.random.seed(42)
    for _ in range(num_neg_points):
        side = np.random.randint(4)
        if side == 0 and y - neg_point_margin > 0:  # top
            neg_points.append([np.clip(np.random.randint(x, x+w), 0, w_img-1), max(0, y - neg_point_margin)])
        elif side == 1 and y + h + neg_point_margin < h_img:  # bottom
            neg_points.append([np.clip(np.random.randint(x, x+w), 0, w_img-1), min(h_img-1, y + h + neg_point_margin)])
        elif side == 2 and x - neg_point_margin > 0:  # left
            neg_points.append([max(0, x - neg_point_margin), np.clip(np.random.randint(y, y+h), 0, h_img-1)])
        elif side == 3 and x + w + neg_point_margin < w_img:  # right
            neg_points.append([min(w_img-1, x + w + neg_point_margin), np.clip(np.random.randint(y, y+h), 0, h_img-1)])
    if neg_points:
        prompts['neg_points'] = np.array(neg_points)
    
    return prompts


def predict_sam2(predictor, image, box, neg_points=None):
    """Run SAM2 prediction with box + optional negative points."""
    import torch
    
    predictor.set_image(image)
    
    point_coords = neg_points if neg_points is not None else None
    point_labels = np.zeros(len(neg_points)) if neg_points is not None else None
    
    device = predictor.model.device
    device_type = str(device).split(':')[0]
    
    # Use appropriate autocast context based on device
    if device_type == "cuda":
        autocast_context = torch.autocast("cuda", dtype=torch.bfloat16)
    elif device_type == "mps":
        autocast_context = torch.autocast("mps", dtype=torch.float16)
    else:
        autocast_context = torch.no_grad()
    
    with autocast_context:
        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=False
        )
    
    return masks[0], scores[0]


def predict_medsam(predictor, image, box):
    """Run MedSAM prediction with box prompt."""
    import torch
    
    predictor.set_image(image)
    
    with torch.no_grad():
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box,
            multimask_output=False
        )
    
    return masks[0], scores[0]


def calculate_dice(pred, gt):
    """Calculate Dice coefficient."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    if pred.sum() + gt.sum() == 0:
        return 1.0
    return (2.0 * intersection) / (pred.sum() + gt.sum())


def create_overlay(image, mask, color, alpha=0.5):
    """Create an overlay of a colored mask on an image."""
    overlay = image.copy()
    mask_bool = mask.astype(bool)
    overlay[mask_bool] = (1 - alpha) * image[mask_bool] + alpha * np.array(color)
    return overlay.astype(np.uint8)


def create_colored_mask(mask, class_colors=CLASS_COLORS):
    """Create RGB colored mask from class mask."""
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in class_colors.items():
        colored[mask == class_id] = color
    return colored


def draw_prompts_on_image(ax, prompts, box_color='lime', point_color='cyan', neg_color='red'):
    """Draw prompt annotations on an axis."""
    if 'box' in prompts:
        box = prompts['box']
        rect = plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                            linewidth=2, edgecolor=box_color, facecolor='none')
        ax.add_patch(rect)
    
    if 'centroid' in prompts:
        cx, cy = prompts['centroid'][0]
        ax.scatter([cx], [cy], c=point_color, s=100, marker='*', edgecolors='white', linewidths=1, zorder=5)
    
    if 'multi_point' in prompts:
        points = prompts['multi_point']
        ax.scatter(points[:, 0], points[:, 1], c=point_color, s=60, marker='o', edgecolors='white', linewidths=0.5, zorder=5)
    
    if 'neg_points' in prompts:
        points = prompts['neg_points']
        ax.scatter(points[:, 0], points[:, 1], c=neg_color, s=60, marker='x', linewidths=2, zorder=5)


def generate_method_comparison_figure(sam2_predictor, medsam_predictor, dataset, output_dir, num_samples=4):
    """
    Figure 1: Side-by-side comparison of methods on multiple test images.
    Shows: Original | GT | SAM2 Centroid | SAM2 Box | SAM2 Box+Neg | MedSAM Box
    """
    print("\n📊 Generating method comparison figure...")
    
    # Select diverse samples
    np.random.seed(42)
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    fig, axes = plt.subplots(num_samples, 6, figsize=(18, 3.5 * num_samples))
    
    methods = ['Original', 'Ground Truth', 'SAM2\nCentroid', 'SAM2\nBox', 'SAM2\nBox+Neg', 'MedSAM\nBox']
    
    for row_idx, sample_idx in enumerate(indices):
        sample = dataset[sample_idx]
        image = sample['image_np']
        gt_mask = sample['mask'].numpy()
        filename = sample['filename']
        
        # Get unique classes (pick one that exists)
        unique_classes = [c for c in sample['unique_classes'] if c in CLASS_NAMES]
        if not unique_classes:
            continue
        target_class = unique_classes[0]  # Pick first available class
        binary_gt = (gt_mask == target_class).astype(np.uint8)
        
        prompts = get_prompts_from_mask(binary_gt)
        if 'box' not in prompts:
            continue
        
        # Predictions
        predictions = {}
        
        # SAM2 Centroid
        if 'centroid' in prompts:
            try:
                sam2_predictor.set_image(image)
                import torch
                device = sam2_predictor.model.device
                device_type = str(device).split(':')[0]
                if device_type == "cuda":
                    context = torch.autocast("cuda", dtype=torch.bfloat16)
                elif device_type == "mps":
                    context = torch.autocast("mps", dtype=torch.float16)
                else:
                    context = torch.no_grad()
                with context:
                    masks, _, _ = sam2_predictor.predict(
                        point_coords=prompts['centroid'],
                        point_labels=np.array([1]),
                        box=None,
                        multimask_output=False
                    )
                predictions['centroid'] = masks[0]
            except Exception as e:
                print(f"    ⚠ Centroid prediction failed: {e}")
                predictions['centroid'] = np.zeros_like(binary_gt)
        else:
            predictions['centroid'] = np.zeros_like(binary_gt)
        
        # SAM2 Box
        predictions['box'], _ = predict_sam2(sam2_predictor, image, prompts['box'], None)
        
        # SAM2 Box + Neg
        neg_pts = prompts.get('neg_points', None)
        predictions['box_neg'], _ = predict_sam2(sam2_predictor, image, prompts['box'], neg_pts)
        
        # MedSAM Box
        if medsam_predictor is not None:
            predictions['medsam'], _ = predict_medsam(medsam_predictor, image, prompts['box'])
        else:
            predictions['medsam'] = np.zeros_like(binary_gt)
        
        # Calculate Dice scores
        dice_scores = {
            'centroid': calculate_dice(predictions['centroid'], binary_gt),
            'box': calculate_dice(predictions['box'], binary_gt),
            'box_neg': calculate_dice(predictions['box_neg'], binary_gt),
            'medsam': calculate_dice(predictions['medsam'], binary_gt),
        }
        
        # Plot row
        class_name = CLASS_NAMES.get(target_class, f'Class {target_class}')
        
        # Original
        axes[row_idx, 0].imshow(image)
        axes[row_idx, 0].set_title(f'{filename[:20]}...\n({class_name})', fontsize=9)
        axes[row_idx, 0].axis('off')
        
        # Ground Truth
        gt_colored = create_overlay(image, binary_gt, CLASS_COLORS[target_class], alpha=0.6)
        axes[row_idx, 1].imshow(gt_colored)
        axes[row_idx, 1].set_title('Ground Truth', fontsize=10)
        axes[row_idx, 1].axis('off')
        
        # SAM2 Centroid
        pred_overlay = create_overlay(image, predictions['centroid'], [30, 144, 255], alpha=0.6)
        axes[row_idx, 2].imshow(pred_overlay)
        if 'centroid' in prompts:
            cx, cy = prompts['centroid'][0]
            axes[row_idx, 2].scatter([cx], [cy], c='lime', s=150, marker='*', edgecolors='white', linewidths=1.5, zorder=5)
        axes[row_idx, 2].set_title(f'Dice: {dice_scores["centroid"]:.3f}', fontsize=10)
        axes[row_idx, 2].axis('off')
        
        # SAM2 Box
        pred_overlay = create_overlay(image, predictions['box'], [30, 144, 255], alpha=0.6)
        axes[row_idx, 3].imshow(pred_overlay)
        box = prompts['box']
        rect = plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                            linewidth=2, edgecolor='lime', facecolor='none')
        axes[row_idx, 3].add_patch(rect)
        axes[row_idx, 3].set_title(f'Dice: {dice_scores["box"]:.3f}', fontsize=10)
        axes[row_idx, 3].axis('off')
        
        # SAM2 Box + Neg
        pred_overlay = create_overlay(image, predictions['box_neg'], [30, 144, 255], alpha=0.6)
        axes[row_idx, 4].imshow(pred_overlay)
        rect = plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                            linewidth=2, edgecolor='lime', facecolor='none')
        axes[row_idx, 4].add_patch(rect)
        if neg_pts is not None:
            axes[row_idx, 4].scatter(neg_pts[:, 0], neg_pts[:, 1], c='red', s=60, marker='x', linewidths=2, zorder=5)
        axes[row_idx, 4].set_title(f'Dice: {dice_scores["box_neg"]:.3f}', fontsize=10)
        axes[row_idx, 4].axis('off')
        
        # MedSAM
        pred_overlay = create_overlay(image, predictions['medsam'], [255, 165, 0], alpha=0.6)
        axes[row_idx, 5].imshow(pred_overlay)
        rect = plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                            linewidth=2, edgecolor='orange', facecolor='none')
        axes[row_idx, 5].add_patch(rect)
        axes[row_idx, 5].set_title(f'Dice: {dice_scores["medsam"]:.3f}', fontsize=10)
        axes[row_idx, 5].axis('off')
    
    # Column headers
    for col_idx, method in enumerate(methods):
        axes[0, col_idx].set_title(f'{method}\n' + axes[0, col_idx].get_title(), fontsize=10, fontweight='bold')
    
    plt.suptitle('Segmentation Method Comparison on BCSS Test Set', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save
    for fmt in ['png', 'pdf']:
        plt.savefig(os.path.join(output_dir, f'qualitative_method_comparison.{fmt}'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved qualitative_method_comparison.png/pdf")


def generate_per_class_figure(sam2_predictor, dataset, output_dir):
    """
    Figure 2: Best model (SAM2 Box+Neg) predictions for each tissue class.
    One row per class showing successful segmentation examples.
    """
    print("\n📊 Generating per-class results figure...")
    
    target_classes = [1, 2, 3, 4, 18]
    class_examples = {c: [] for c in target_classes}
    
    # Collect examples for each class
    for idx in range(len(dataset)):
        sample = dataset[idx]
        gt_mask = sample['mask'].numpy()
        unique = sample['unique_classes']
        
        for c in target_classes:
            if c in unique and len(class_examples[c]) < 3:
                class_examples[c].append(idx)
    
    fig, axes = plt.subplots(5, 4, figsize=(14, 16))
    
    for row_idx, class_id in enumerate(target_classes):
        if not class_examples[class_id]:
            continue
        
        sample_idx = class_examples[class_id][0]
        sample = dataset[sample_idx]
        image = sample['image_np']
        gt_mask = sample['mask'].numpy()
        binary_gt = (gt_mask == class_id).astype(np.uint8)
        
        prompts = get_prompts_from_mask(binary_gt)
        if 'box' not in prompts:
            continue
        
        # Best method prediction (Box + Neg)
        neg_pts = prompts.get('neg_points', None)
        pred_mask, _ = predict_sam2(sam2_predictor, image, prompts['box'], neg_pts)
        dice = calculate_dice(pred_mask, binary_gt)
        
        class_name = CLASS_NAMES[class_id]
        color = CLASS_COLORS[class_id]
        
        # Original
        axes[row_idx, 0].imshow(image)
        axes[row_idx, 0].set_ylabel(class_name, fontsize=12, fontweight='bold', rotation=0, labelpad=60, va='center')
        axes[row_idx, 0].set_title('Original' if row_idx == 0 else '', fontsize=11)
        axes[row_idx, 0].axis('off')
        
        # GT overlay
        gt_overlay = create_overlay(image, binary_gt, color, alpha=0.6)
        axes[row_idx, 1].imshow(gt_overlay)
        axes[row_idx, 1].set_title('Ground Truth' if row_idx == 0 else '', fontsize=11)
        axes[row_idx, 1].axis('off')
        
        # Prediction overlay with prompts
        pred_overlay = create_overlay(image, pred_mask, [30, 144, 255], alpha=0.6)
        axes[row_idx, 2].imshow(pred_overlay)
        box = prompts['box']
        rect = plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                            linewidth=2, edgecolor='lime', facecolor='none')
        axes[row_idx, 2].add_patch(rect)
        if neg_pts is not None:
            axes[row_idx, 2].scatter(neg_pts[:, 0], neg_pts[:, 1], c='red', s=40, marker='x', linewidths=1.5, zorder=5)
        axes[row_idx, 2].set_title('SAM2 Box+Neg' if row_idx == 0 else '', fontsize=11)
        axes[row_idx, 2].axis('off')
        
        # Dice score annotation
        axes[row_idx, 3].text(0.5, 0.5, f'Dice\n{dice:.3f}', 
                             fontsize=24, fontweight='bold', ha='center', va='center',
                             color='darkgreen' if dice > 0.5 else 'darkorange')
        axes[row_idx, 3].axis('off')
        axes[row_idx, 3].set_facecolor('#f0f0f0')
    
    plt.suptitle('SAM2 Box+Neg Segmentation by Tissue Class', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    for fmt in ['png', 'pdf']:
        plt.savefig(os.path.join(output_dir, f'qualitative_per_class.{fmt}'),
                   dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved qualitative_per_class.png/pdf")


def generate_success_failure_figure(sam2_predictor, dataset, output_dir):
    """
    Figure 3: Success vs Failure cases to show model limitations.
    Top row: High Dice (>0.7), Bottom row: Low Dice (<0.4)
    """
    print("\n📊 Generating success/failure cases figure...")
    
    results = []
    
    for idx in range(min(len(dataset), 45)):  # Check first 45 samples
        sample = dataset[idx]
        image = sample['image_np']
        gt_mask = sample['mask'].numpy()
        unique = sample['unique_classes']
        
        for class_id in unique:
            if class_id not in CLASS_NAMES or class_id == 0:
                continue
            
            binary_gt = (gt_mask == class_id).astype(np.uint8)
            prompts = get_prompts_from_mask(binary_gt)
            if 'box' not in prompts:
                continue
            
            neg_pts = prompts.get('neg_points', None)
            pred_mask, _ = predict_sam2(sam2_predictor, image, prompts['box'], neg_pts)
            dice = calculate_dice(pred_mask, binary_gt)
            
            results.append({
                'idx': idx,
                'class_id': class_id,
                'dice': dice,
                'image': image,
                'gt': binary_gt,
                'pred': pred_mask,
                'prompts': prompts,
                'filename': sample['filename']
            })
    
    # Sort by Dice
    results.sort(key=lambda x: x['dice'], reverse=True)
    
    # Get top 3 successes and bottom 3 failures
    successes = results[:3]
    failures = [r for r in results if r['dice'] < 0.4][:3]
    
    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
    
    for col_idx, res in enumerate(successes):
        # Success row
        image = res['image']
        pred_overlay = create_overlay(image, res['pred'], [30, 144, 255], alpha=0.6)
        gt_overlay = create_overlay(image, res['gt'], CLASS_COLORS[res['class_id']], alpha=0.6)
        
        axes[0, col_idx*2].imshow(gt_overlay)
        axes[0, col_idx*2].set_title(f'GT: {CLASS_NAMES[res["class_id"]]}', fontsize=10)
        axes[0, col_idx*2].axis('off')
        
        axes[0, col_idx*2+1].imshow(pred_overlay)
        box = res['prompts']['box']
        rect = plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                            linewidth=2, edgecolor='lime', facecolor='none')
        axes[0, col_idx*2+1].add_patch(rect)
        axes[0, col_idx*2+1].set_title(f'Dice: {res["dice"]:.3f} ✓', fontsize=10, color='darkgreen')
        axes[0, col_idx*2+1].axis('off')
    
    for col_idx, res in enumerate(failures):
        if col_idx >= 3:
            break
        # Failure row
        image = res['image']
        pred_overlay = create_overlay(image, res['pred'], [30, 144, 255], alpha=0.6)
        gt_overlay = create_overlay(image, res['gt'], CLASS_COLORS[res['class_id']], alpha=0.6)
        
        axes[1, col_idx*2].imshow(gt_overlay)
        axes[1, col_idx*2].set_title(f'GT: {CLASS_NAMES[res["class_id"]]}', fontsize=10)
        axes[1, col_idx*2].axis('off')
        
        axes[1, col_idx*2+1].imshow(pred_overlay)
        box = res['prompts']['box']
        rect = plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                            linewidth=2, edgecolor='red', facecolor='none')
        axes[1, col_idx*2+1].add_patch(rect)
        axes[1, col_idx*2+1].set_title(f'Dice: {res["dice"]:.3f} ✗', fontsize=10, color='darkred')
        axes[1, col_idx*2+1].axis('off')
    
    # Row labels
    axes[0, 0].set_ylabel('Success Cases\n(Dice > 0.7)', fontsize=11, fontweight='bold', rotation=0, labelpad=80, va='center')
    axes[1, 0].set_ylabel('Failure Cases\n(Dice < 0.4)', fontsize=11, fontweight='bold', rotation=0, labelpad=80, va='center')
    
    plt.suptitle('SAM2 Box+Neg: Success vs Failure Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    for fmt in ['png', 'pdf']:
        plt.savefig(os.path.join(output_dir, f'qualitative_success_failure.{fmt}'),
                   dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved qualitative_success_failure.png/pdf")


def generate_prompt_comparison_figure(sam2_predictor, dataset, output_dir):
    """
    Figure 4: Visual comparison of different prompt strategies on same image.
    Shows how prompts affect segmentation quality.
    """
    print("\n📊 Generating prompt comparison figure...")
    
    # Find a good example image
    sample = dataset[5]
    image = sample['image_np']
    gt_mask = sample['mask'].numpy()
    
    # Pick a class that exists
    unique = sample['unique_classes']
    target_class = 1 if 1 in unique else unique[0]
    binary_gt = (gt_mask == target_class).astype(np.uint8)
    prompts = get_prompts_from_mask(binary_gt)
    
    if 'box' not in prompts:
        print("  ⚠ Could not generate prompts for selected sample")
        return
    
    import torch
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Row 1: Prompts visualization
    # Original
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image', fontsize=11, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Centroid prompt
    axes[0, 1].imshow(image)
    if 'centroid' in prompts:
        cx, cy = prompts['centroid'][0]
        axes[0, 1].scatter([cx], [cy], c='lime', s=300, marker='*', edgecolors='white', linewidths=2, zorder=5)
    axes[0, 1].set_title('Centroid Prompt', fontsize=11, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Box prompt
    axes[0, 2].imshow(image)
    box = prompts['box']
    rect = plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                        linewidth=3, edgecolor='lime', facecolor='none')
    axes[0, 2].add_patch(rect)
    axes[0, 2].set_title('Box Prompt', fontsize=11, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Box + Neg prompt
    axes[0, 3].imshow(image)
    rect = plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                        linewidth=3, edgecolor='lime', facecolor='none')
    axes[0, 3].add_patch(rect)
    if 'neg_points' in prompts:
        neg_pts = prompts['neg_points']
        axes[0, 3].scatter(neg_pts[:, 0], neg_pts[:, 1], c='red', s=100, marker='x', linewidths=3, zorder=5)
    axes[0, 3].set_title('Box + Negative Points', fontsize=11, fontweight='bold')
    axes[0, 3].axis('off')
    
    # Row 2: Predictions
    # Ground Truth
    gt_overlay = create_overlay(image, binary_gt, CLASS_COLORS[target_class], alpha=0.6)
    axes[1, 0].imshow(gt_overlay)
    axes[1, 0].set_title(f'Ground Truth\n({CLASS_NAMES[target_class]})', fontsize=11)
    axes[1, 0].axis('off')
    
    # Centroid prediction
    sam2_predictor.set_image(image)
    if 'centroid' in prompts:
        device = sam2_predictor.model.device
        device_type = str(device).split(':')[0]
        if device_type == "cuda":
            context = torch.autocast("cuda", dtype=torch.bfloat16)
        elif device_type == "mps":
            context = torch.autocast("mps", dtype=torch.float16)
        else:
            context = torch.no_grad()
        with context:
            masks, _, _ = sam2_predictor.predict(
                point_coords=prompts['centroid'],
                point_labels=np.array([1]),
                box=None,
                multimask_output=False
            )
        centroid_pred = masks[0]
        dice_centroid = calculate_dice(centroid_pred, binary_gt)
        pred_overlay = create_overlay(image, centroid_pred, [30, 144, 255], alpha=0.6)
        axes[1, 1].imshow(pred_overlay)
        axes[1, 1].set_title(f'Dice: {dice_centroid:.3f}', fontsize=11, 
                            color='darkgreen' if dice_centroid > 0.5 else 'darkorange')
    else:
        axes[1, 1].imshow(image)
        axes[1, 1].set_title('N/A', fontsize=11)
    axes[1, 1].axis('off')
    
    # Box prediction
    box_pred, _ = predict_sam2(sam2_predictor, image, prompts['box'], None)
    dice_box = calculate_dice(box_pred, binary_gt)
    pred_overlay = create_overlay(image, box_pred, [30, 144, 255], alpha=0.6)
    axes[1, 2].imshow(pred_overlay)
    axes[1, 2].set_title(f'Dice: {dice_box:.3f}', fontsize=11,
                        color='darkgreen' if dice_box > 0.5 else 'darkorange')
    axes[1, 2].axis('off')
    
    # Box + Neg prediction
    neg_pts = prompts.get('neg_points', None)
    box_neg_pred, _ = predict_sam2(sam2_predictor, image, prompts['box'], neg_pts)
    dice_box_neg = calculate_dice(box_neg_pred, binary_gt)
    pred_overlay = create_overlay(image, box_neg_pred, [30, 144, 255], alpha=0.6)
    axes[1, 3].imshow(pred_overlay)
    axes[1, 3].set_title(f'Dice: {dice_box_neg:.3f}', fontsize=11,
                        color='darkgreen' if dice_box_neg > 0.5 else 'darkorange')
    axes[1, 3].axis('off')
    
    plt.suptitle('Impact of Prompt Strategy on Segmentation Quality', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    for fmt in ['png', 'pdf']:
        plt.savefig(os.path.join(output_dir, f'qualitative_prompt_comparison.{fmt}'),
                   dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved qualitative_prompt_comparison.png/pdf")


def generate_full_image_segmentation(sam2_predictor, dataset, output_dir):
    """
    Figure 5: Full multi-class segmentation visualization.
    Shows complete image with all classes segmented and colored.
    """
    print("\n📊 Generating full image segmentation figure...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for fig_idx in range(2):
        sample = dataset[fig_idx * 10 + 3]  # Pick diverse samples
        image = sample['image_np']
        gt_mask = sample['mask'].numpy()
        filename = sample['filename']
        
        # Create full predicted mask
        pred_full = np.zeros_like(gt_mask)
        
        for class_id in [1, 2, 3, 4, 18]:
            binary_gt = (gt_mask == class_id).astype(np.uint8)
            if binary_gt.sum() == 0:
                continue
            
            prompts = get_prompts_from_mask(binary_gt)
            if 'box' not in prompts:
                continue
            
            neg_pts = prompts.get('neg_points', None)
            pred_mask, _ = predict_sam2(sam2_predictor, image, prompts['box'], neg_pts)
            pred_full[pred_mask > 0.5] = class_id
        
        # Create colored masks
        gt_colored = create_colored_mask(gt_mask)
        pred_colored = create_colored_mask(pred_full)
        
        # Plot
        axes[fig_idx, 0].imshow(image)
        axes[fig_idx, 0].set_title(f'Original: {filename[:25]}...', fontsize=10)
        axes[fig_idx, 0].axis('off')
        
        axes[fig_idx, 1].imshow(gt_colored)
        axes[fig_idx, 1].set_title('Ground Truth (All Classes)', fontsize=10)
        axes[fig_idx, 1].axis('off')
        
        axes[fig_idx, 2].imshow(pred_colored)
        axes[fig_idx, 2].set_title('SAM2 Prediction (All Classes)', fontsize=10)
        axes[fig_idx, 2].axis('off')
    
    # Add legend
    legend_elements = [mpatches.Patch(facecolor=np.array(color)/255, 
                                      label=CLASS_NAMES[cid])
                      for cid, color in CLASS_COLORS.items() if cid != 0]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=10,
              bbox_to_anchor=(0.5, -0.02))
    
    plt.suptitle('Full Multi-Class Histopathology Segmentation', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    for fmt in ['png', 'pdf']:
        plt.savefig(os.path.join(output_dir, f'qualitative_full_segmentation.{fmt}'),
                   dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved qualitative_full_segmentation.png/pdf")


def main():
    """Generate all qualitative result figures."""
    print("=" * 60)
    print("GENERATING QUALITATIVE RESULTS FIGURES")
    print("=" * 60)
    
    # Setup output directory
    output_dir = os.path.join(project_root, 'results', 'figures', 'qualitative')
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Load dataset
    from dataset import BCSSDataset
    image_dir = os.path.join(project_root, 'data/bcss/images')
    mask_dir = os.path.join(project_root, 'data/bcss/masks')
    dataset = BCSSDataset(image_dir, mask_dir, split='test')
    print(f"Loaded {len(dataset)} test samples")
    
    # Load models
    sam2_predictor, medsam_predictor, device = load_models()
    
    # Generate figures
    generate_method_comparison_figure(sam2_predictor, medsam_predictor, dataset, output_dir, num_samples=4)
    generate_per_class_figure(sam2_predictor, dataset, output_dir)
    generate_prompt_comparison_figure(sam2_predictor, dataset, output_dir)
    generate_success_failure_figure(sam2_predictor, dataset, output_dir)
    generate_full_image_segmentation(sam2_predictor, dataset, output_dir)
    
    print("\n" + "=" * 60)
    print("✅ ALL QUALITATIVE FIGURES GENERATED!")
    print(f"📁 Saved to: {output_dir}")
    print("=" * 60)
    
    # List generated files
    print("\nGenerated files:")
    for f in sorted(os.listdir(output_dir)):
        print(f"  - {f}")


if __name__ == '__main__':
    main()
