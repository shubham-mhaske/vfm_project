import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import cv2

# Add the sam2 directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'sam2')))

import sam2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from dataset import BCSSDataset

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0]
    w, h = box[1] - box[0]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def get_sam2_predictor(model_cfg, checkpoint, device):
    """Initializes and returns a SAM 2 predictor on the specified device."""
    # Get the path to the sam2 package root
    sam2_pkg_root = sam2.__path__[0]
    # Get the relative path of the config file w.r.t. the package root
    relative_cfg_path = os.path.relpath(model_cfg, sam2_pkg_root)
    # Hydra expects forward slashes
    relative_cfg_path = relative_cfg_path.replace('\\', '/')

    # The config name should be relative to the package's config directory,
    # which is what build_sam2 expects.
    predictor = SAM2ImagePredictor(build_sam2(relative_cfg_path, checkpoint, device=device))

    return predictor

def get_prompts_from_mask(binary_mask, num_points=5, neg_point_margin=10):
    """Generates various prompts from a binary mask."""
    prompts = {}
    
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return prompts

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    prompts['box'] = np.array([[x, y], [x + w, y + h]])

    M = cv2.moments(largest_contour)
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        prompts['centroid'] = (np.array([[cx, cy]]), np.array([1]))

    if len(largest_contour) >= num_points:
        step = len(largest_contour) // num_points
        points = largest_contour[::step, 0, :]
        prompts['multi_point'] = (points, np.ones(len(points)))

    # Negative points outside the bounding box
    neg_points = []
    for _ in range(num_points):
        side = np.random.randint(4)
        if side == 0: # top
            neg_points.append([np.random.randint(x, x+w), y - neg_point_margin])
        elif side == 1: # bottom
            neg_points.append([np.random.randint(x, x+w), y + h + neg_point_margin])
        elif side == 2: # left
            neg_points.append([x - neg_point_margin, np.random.randint(y, y+h)])
        else: # right
            neg_points.append([x + w + neg_point_margin, np.random.randint(y, y+h)])
    prompts['neg_points'] = (np.array(neg_points), np.zeros(len(neg_points)))
        
    return prompts

def postprocess_mask(mask, kernel_size=7, min_area_ratio=0.1):
    """Cleans up a binary mask using morphological operations and contour filtering."""
    # Closing to fill small holes
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed_mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    # Keep only the largest connected component
    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return closed_mask
    
    largest_contour = max(contours, key=cv2.contourArea)
    final_mask = np.zeros_like(mask)
    cv2.drawContours(final_mask, [largest_contour], -1, 1, thickness=cv2.FILLED)
    
    return final_mask

def get_predicted_mask_from_prompts(predictor, image, prompts, prompt_type='centroid', use_neg_points=False):
    """Gets the predicted mask from SAM 2 using various prompts."""
    predictor.set_image(image)
    
    point_coords, point_labels, box = None, None, None
    
    if prompt_type == 'centroid':
        point_coords, point_labels = prompts['centroid']
    elif prompt_type == 'multi_point':
        point_coords, point_labels = prompts['multi_point']
    elif prompt_type == 'box':
        box = prompts['box']

    if use_neg_points and 'neg_points' in prompts:
        neg_coords, neg_labels = prompts['neg_points']
        if point_coords is not None:
            point_coords = np.concatenate([point_coords, neg_coords], axis=0)
            point_labels = np.concatenate([point_labels, neg_labels], axis=0)
        else:
            point_coords, point_labels = neg_coords, neg_labels
    
    device = predictor.model.device
    if device == "mps":
        autocast_context = torch.autocast("mps", dtype=torch.float16)
    elif device == "cuda":
        autocast_context = torch.autocast("cuda", dtype=torch.bfloat16)
    else:
        autocast_context = torch.no_grad() # No autocast for CPU, just no_grad

    with autocast_context:
        masks, _, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=False
        )
    
    return masks[0], point_coords, point_labels

def get_predicted_mask(predictor, image, gt_mask, prompt_type='centroid', use_neg_points=False):
    """
    Generates a predicted mask for the entire foreground.
    """
    # 1. Create a binary ground truth mask for the foreground
    binary_gt_mask = (gt_mask > 0).astype(np.uint8)

    # 2. Generate prompts from this combined foreground mask.
    prompts = get_prompts_from_mask(binary_gt_mask)

    predicted_mask = np.zeros_like(binary_gt_mask)
    points = None
    labels = None

    if prompt_type in prompts:
        # 3. Get a single predicted mask for the foreground.
        predicted_mask, points, labels = get_predicted_mask_from_prompts(
            predictor,
            image,
            prompts,
            prompt_type=prompt_type,
            use_neg_points=use_neg_points
        )

    # 4. Return the predicted mask and the binary ground truth mask.
    return predicted_mask, binary_gt_mask, points, labels

def calculate_metrics(pred_mask, gt_mask):
    """Calculates Dice Similarity Coefficient and Intersection over Union."""
    pred_mask = pred_mask.astype(np.bool_)
    gt_mask = gt_mask.astype(np.bool_)
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    dice = (2. * intersection) / (pred_mask.sum() + gt_mask.sum()) if (pred_mask.sum() + gt_mask.sum()) > 0 else 0
    iou = intersection / np.logical_or(pred_mask, gt_mask).sum() if np.logical_or(pred_mask, gt_mask).sum() > 0 else 0
    return dice, iou

def run_and_visualize(predictor, image, gt_mask, class_id, prompt_type, use_neg_points=False, apply_postprocessing=False):
    """Runs segmentation for a given prompt type and visualizes the result."""
    binary_mask = (gt_mask == class_id).astype(np.uint8)
    prompts = get_prompts_from_mask(binary_mask)

    if prompt_type not in prompts:
        print(f"Prompt type '{prompt_type}' could not be generated for this sample.")
        return

    predicted_mask, points, labels = get_predicted_mask_from_prompts(predictor, image, prompts, prompt_type, use_neg_points)
    
    if apply_postprocessing:
        predicted_mask = postprocess_mask(predicted_mask)

    dice, iou = calculate_metrics(predicted_mask, binary_mask)
    
    config_name = f"{prompt_type}"
    if use_neg_points:
        config_name += "+neg_points"
    if apply_postprocessing:
        config_name += "+postprocessed"

    print(f"Results for {config_name}: Dice={dice:.4f}, IoU={iou:.4f}")

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(predicted_mask, plt.gca())
    show_mask(binary_mask, plt.gca(), random_color=True)
    
    if points is not None:
        show_points(points, labels, plt.gca())
    if prompt_type == 'box':
        show_box(prompts['box'], plt.gca())

    plt.title(f"SAM 2 with {config_name} | Dice: {dice:.4f}, IoU: {iou:.4f}")
    plt.axis('off')
    output_filename = f"sam2_segmentation_{config_name}.png"
    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
    print(f"Result saved to {output_filename}")
    plt.close()

def main(args):
    """Main function to run the segmentation with different prompt strategies."""
    # --- 0. Device Selection ---
    device = get_device()
    print(f"Using device: {device}")

    # Load dataset
    bcss_dataset = BCSSDataset(image_dir=args.image_dir, mask_dir=args.mask_dir, split='test')
    sample = bcss_dataset[0]
    image = sample['image_np']
    gt_mask = sample['mask'].numpy()
    class_id = args.class_id # Example class_id

    # Get SAM 2 predictor
    predictor = get_sam2_predictor(args.model_cfg, args.checkpoint, device)

    # Run for different prompt types and enhancements
    run_and_visualize(predictor, image, gt_mask, class_id, 'box', use_neg_points=False, apply_postprocessing=False)
    run_and_visualize(predictor, image, gt_mask, class_id, 'box', use_neg_points=True, apply_postprocessing=False)
    run_and_visualize(predictor, image, gt_mask, class_id, 'box', use_neg_points=True, apply_postprocessing=True)

if __name__ == '__main__':
    import argparse
    from device_utils import get_device

    parser = argparse.ArgumentParser(description="Run SAM 2 segmentation with different prompt strategies.")
    parser.add_argument('--image_dir', type=str, default='data/bcss/images', help='Directory for images.')
    parser.add_argument('--mask_dir', type=str, default='data/bcss/masks', help='Directory for masks.')
    parser.add_argument('--model_cfg', type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml", help='SAM model configuration file.')
    parser.add_argument('--checkpoint', type=str, default="sam2/checkpoints/sam2.1_hiera_large.pt", help='Path to SAM checkpoint.')
    parser.add_argument('--class_id', type=int, default=1, help='Class ID to segment.')
    args = parser.parse_args()
    main(args)