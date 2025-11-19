import torch
import numpy as np
from tqdm import tqdm
from dataset import BCSSDataset
from sam_segmentation import get_sam2_predictor, get_predicted_mask, calculate_metrics
import argparse
from device_utils import get_device

def evaluate_segmentation(args):
    """Runs the SAM 2 segmentation and evaluation on the full test set."""
    # Load the dataset
    image_dir = '../data/bcss/images'
    mask_dir = '../data/bcss/masks'
    bcss_dataset = BCSSDataset(image_dir=image_dir, mask_dir=mask_dir, split='test')

    # --- Device Selection ---
    device = get_device()
    print(f"Using device: {device}")

    # Get SAM 2 predictor
    model_cfg = "../sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
    predictor = get_sam2_predictor(model_cfg, args.checkpoint, device)

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
    parser = argparse.ArgumentParser(description="Evaluate SAM 2 segmentation.")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to SAM checkpoint.')
    args = parser.parse_args()
    evaluate_segmentation(args)
