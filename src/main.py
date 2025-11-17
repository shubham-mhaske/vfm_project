import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import argparse

# Add the sam2 directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'sam2')))

from dataset import BCSSDataset
from sam_segmentation import (
    get_sam2_predictor,
    get_prompts_from_mask,
    get_predicted_mask_from_prompts,
    postprocess_mask,
    calculate_metrics,
    show_mask,
    show_points,
    show_box
)
from clip_classification import (
    CLIPClassifier,
    crop_region_from_mask,
    load_prompts_from_json
)
from device_utils import device, get_device

def main(args):
    """Main function to run the end-to-end segmentation and classification pipeline."""
    # --- 0. Device Selection ---
    device = get_device()
    print(f"Using device: {device}")

    # --- 1. Load Data ---
    bcss_dataset = BCSSDataset(image_dir=args.image_dir, mask_dir=args.mask_dir, split='test')
    sample = bcss_dataset[args.sample_id]
    image = sample['image_np']
    gt_mask = sample['mask'].numpy()
    class_id = 1 # We are targeting the 'tumor' class for this sample
    gt_class_name = bcss_dataset.class_names[class_id]

    # --- 2. Run Segmentation ---
    print("Running SAM segmentation...")
    # Initialize predictor
    sam_predictor = get_sam2_predictor(args.sam_model_cfg, args.sam_checkpoint, device=device)

    # Get prompts and predict mask
    binary_gt_mask = (gt_mask == class_id).astype(np.uint8)
    prompts = get_prompts_from_mask(binary_gt_mask)
    predicted_mask, _, _ = get_predicted_mask_from_prompts(sam_predictor, image, prompts, prompt_type='box', use_neg_points=True)
    
    # Post-process the mask
    final_mask = postprocess_mask(predicted_mask)

    seg_dice, seg_iou = calculate_metrics(final_mask, binary_gt_mask)
    print(f"Segmentation Performance: Dice={seg_dice:.4f}, IoU={seg_iou:.4f}")

    # --- 3. Run Classification ---
    print("\nRunning CLIP classification...")
    # Initialize classifier
    clip_classifier = CLIPClassifier(device=device)
    
    # Load prompts
    clip_prompts = load_prompts_from_json(args.prompt_file)

    # Crop the segmented region
    cropped_image = crop_region_from_mask(image, final_mask)

    # Classify the region
    predicted_class_name = clip_classifier.classify_region(cropped_image, clip_prompts)

    print(f"Ground Truth Class: {gt_class_name}")
    print(f"Predicted Class: {predicted_class_name}")

    # --- 4. Visualize Result ---
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    # Panel 1: Original Image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("1. Original Image")
    axes[0, 0].axis('off')

    # Panel 2: Ground Truth
    axes[0, 1].imshow(image)
    show_mask(binary_gt_mask, axes[0, 1], random_color=True)
    axes[0, 1].set_title(f"2. Ground Truth: '{gt_class_name}'")
    axes[0, 1].axis('off')

    # Panel 3: Our Prediction
    axes[1, 0].imshow(image)
    show_mask(final_mask, axes[1, 0])
    axes[1, 0].set_title(f"3. Our Prediction (Dice: {seg_dice:.4f})")
    axes[1, 0].axis('off')

    # Panel 4: Cropped for CLIP
    if cropped_image:
        axes[1, 1].imshow(cropped_image)
    axes[1, 1].set_title(f"4. Input to CLIP -> Predicted: '{predicted_class_name}'")
    axes[1, 1].axis('off')

    fig.suptitle("End-to-End Zero-Shot Segmentation and Classification", fontsize=16)
    plt.tight_layout()
    output_filename = os.path.join(args.output_dir, "end_to_end_result_detailed.png")
    os.makedirs(args.output_dir, exist_ok=True)
    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
    print(f"\nDetailed result saved to {output_filename}")
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="End-to-end segmentation and classification pipeline.")
    parser.add_argument('--image_dir', type=str, default='data/bcss/images', help='Directory containing images.')
    parser.add_argument('--mask_dir', type=str, default='data/bcss/masks', help='Directory containing masks.')
    parser.add_argument('--output_dir', type=str, default='results/main', help='Directory to save output.')
    parser.add_argument('--sample_id', type=int, default=10, help='Sample ID to process.')
    parser.add_argument('--sam_model_cfg', type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml", help='SAM model configuration file.')
    parser.add_argument('--sam_checkpoint', type=str, default="sam2/checkpoints/sam2.1_hiera_large.pt", help='Path to SAM checkpoint.')
    parser.add_argument('--prompt_file', type=str, required=True, help='Path to the JSON file with CLIP prompts.')
    args = parser.parse_args()
    main(args)
