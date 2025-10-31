import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
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
    calculate_metrics
)
from clip_classification import (
    CLIPClassifier,
    crop_region_from_mask
)

def run_evaluation():
    """Runs the full quantitative evaluation on the test set."""
    print("Starting quantitative evaluation...")

    # --- 1. Load Data and Models ---
    image_dir = 'data/bcss/images'
    mask_dir = 'data/bcss/masks'
    bcss_dataset = BCSSDataset(image_dir=image_dir, mask_dir=mask_dir, split='test')
    bcss_dataset.image_files = bcss_dataset.image_files[:10]
    
    print("Loading models...")
    sam_model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam_checkpoint = "sam2/checkpoints/sam2.1_hiera_large.pt"
    sam_predictor = get_sam2_predictor(sam_model_cfg, sam_checkpoint)
    clip_classifier = CLIPClassifier()

    # --- 2. Evaluation Loop ---
    segmentation_results = []
    classification_results = []

    print(f"Evaluating on {len(bcss_dataset)} test samples...")
    for sample in tqdm(bcss_dataset):
        image = sample['image_np']
        gt_mask = sample['mask'].numpy()
        
        unique_classes = sample['unique_classes']
        unique_classes = unique_classes[unique_classes != 0] # Exclude background

        for class_id in unique_classes:
            if class_id not in bcss_dataset.class_names:
                continue
            gt_class_name = bcss_dataset.class_names[class_id]
            
            # --- Run Segmentation ---
            binary_gt_mask = (gt_mask == class_id).astype(np.uint8)
            if binary_gt_mask.sum() == 0:
                continue

            prompts = get_prompts_from_mask(binary_gt_mask)
            if not prompts:
                continue

            predicted_mask, _, _ = get_predicted_mask_from_prompts(sam_predictor, image, prompts, prompt_type='box', use_neg_points=True)
            final_mask = postprocess_mask(predicted_mask)
            
            seg_dice, seg_iou = calculate_metrics(final_mask, binary_gt_mask)
            segmentation_results.append({'dice': seg_dice, 'iou': seg_iou})

            # --- Run Classification ---
            cropped_image = crop_region_from_mask(image, final_mask)
            predicted_class_name = clip_classifier.classify_region(cropped_image)

            if predicted_class_name is not None:
                classification_results.append({'true': gt_class_name, 'pred': predicted_class_name})

    # --- 3. Compute and Print Metrics ---
    print("\n--- Evaluation Results ---")

    # Segmentation Metrics
    avg_dice = np.mean([res['dice'] for res in segmentation_results])
    avg_iou = np.mean([res['iou'] for res in segmentation_results])
    print(f"\n[Segmentation]\n")
    print(f"Average Dice Score: {avg_dice:.4f}")
    print(f"Average IoU Score:  {avg_iou:.4f}")

    # Classification Metrics
    true_labels = [res['true'] for res in classification_results]
    pred_labels = [res['pred'] for res in classification_results]
    
    accuracy = accuracy_score(true_labels, pred_labels)
    class_names = list(bcss_dataset.class_names.values())[1:] # Exclude background
    cm = confusion_matrix(true_labels, pred_labels, labels=class_names)

    print(f"\n[Classification]\n")
    print(f"Overall Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(cm)

    # Plot and save confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, xticks_rotation='vertical')
    plt.title("Zero-Shot Classification Confusion Matrix")
    plt.tight_layout()
    cm_filename = "confusion_matrix.png"
    plt.savefig(cm_filename)
    print(f"\nConfusion matrix saved to {cm_filename}")
    plt.close()

if __name__ == '__main__':
    run_evaluation()