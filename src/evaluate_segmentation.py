import torch
import numpy as np
from tqdm import tqdm
from dataset import BCSSDataset
from sam_segmentation import get_sam2_predictor, get_predicted_mask, calculate_metrics

def evaluate_segmentation():
    """Runs the SAM 2 segmentation and evaluation on the full test set."""
    # Load the dataset
    image_dir = '../data/bcss/images'
    mask_dir = '../data/bcss/masks'
    bcss_dataset = BCSSDataset(image_dir=image_dir, mask_dir=mask_dir, split='test')

    # Get SAM 2 predictor
    model_cfg = "../sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
    checkpoint = "../sam2/checkpoints/sam2.1_hiera_large.pt"
    predictor = get_sam2_predictor(model_cfg, checkpoint)

    # Initialize metrics
    total_dice = 0
    total_iou = 0
    num_samples = len(bcss_dataset)

    for i in tqdm(range(num_samples)):
        sample = bcss_dataset[i]
        image = sample['image_np']
        gt_mask = sample['mask'].numpy()

        # Get predicted mask
        predicted_mask, binary_mask, _, _ = get_predicted_mask(predictor, image, gt_mask)

        # Calculate metrics
        dice, iou = calculate_metrics(predicted_mask, binary_mask)
        total_dice += dice
        total_iou += iou

    # Calculate average metrics
    avg_dice = total_dice / num_samples
    avg_iou = total_iou / num_samples

    print(f"Average Dice Score: {avg_dice:.4f}")
    print(f"Average IoU: {avg_iou:.4f}")

if __name__ == '__main__':
    evaluate_segmentation()
