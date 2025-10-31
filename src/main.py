import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

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
    crop_region_from_mask
)

def main():
    """Main function to run the end-to-end segmentation and classification pipeline."""
    # --- 1. Load Data ---
    image_dir = 'data/bcss/images'
    mask_dir = 'data/bcss/masks'
    bcss_dataset = BCSSDataset(image_dir=image_dir, mask_dir=mask_dir, split='test')
    sample = bcss_dataset[10]
    image = sample['image_np']
    gt_mask = sample['mask'].numpy()
    class_id = 1 # We are targeting the 'tumor' class for this sample
    gt_class_name = bcss_dataset.class_names[class_id]

    # --- 2. Run Segmentation ---
    print("Running SAM segmentation...")
    # Initialize predictor
    sam_model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam_checkpoint = "sam2/checkpoints/sam2.1_hiera_large.pt"
    sam_predictor = get_sam2_predictor(sam_model_cfg, sam_checkpoint)

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
    clip_classifier = CLIPClassifier()

    # Crop the segmented region
    cropped_image = crop_region_from_mask(image, final_mask)

    # Classify the region
    predicted_class_name = clip_classifier.classify_region(cropped_image)

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
    output_filename = "end_to_end_result_detailed.png"
    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
    print(f"\nDetailed result saved to {output_filename}")
    plt.close()

if __name__ == '__main__':
    main()
