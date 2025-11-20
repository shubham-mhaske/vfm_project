import os
import argparse
import torch
import numpy as np
import json
from tqdm import tqdm
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from src.dataset import BCSSDataset
from src.sam_segmentation import get_prompts_from_mask, get_predicted_mask_from_prompts, calculate_metrics

def evaluate_checkpoint(args):
    print(f"Loading model from {args.checkpoint} with config {args.config}...")
    
    # Load model
    sam2_model = build_sam2(args.config, args.checkpoint, device=args.device)
    predictor = SAM2ImagePredictor(sam2_model)
    
    # Load dataset
    dataset = BCSSDataset(args.data_dir, split='test')
    print(f"Evaluating on {len(dataset)} images...")
    
    all_metrics = []
    
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        image = sample['image_np']
        gt_mask = sample['mask']
        filename = sample['filename']
        
        predictor.set_image(image)
        
        # Get unique objects in GT
        obj_ids = np.unique(gt_mask)
        obj_ids = obj_ids[obj_ids != 0]
        
        image_metrics = []
        
        for obj_id in obj_ids:
            binary_mask = (gt_mask == obj_id).astype(np.uint8)
            
            # Generate prompts (box)
            prompts = get_prompts_from_mask(binary_mask, prompt_type='box')
            
            if 'box' not in prompts:
                continue
                
            box = prompts['box']
            
            # Predict
            masks, scores, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box,
                multimask_output=False
            )
            
            pred_mask = masks[0]
            
            # Calculate metrics
            dice, iou = calculate_metrics(pred_mask, binary_mask)
            image_metrics.append({'dice': dice, 'iou': iou})
            
        if image_metrics:
            avg_dice = np.mean([m['dice'] for m in image_metrics])
            avg_iou = np.mean([m['iou'] for m in image_metrics])
            all_metrics.append({
                'filename': filename,
                'dice': avg_dice,
                'iou': avg_iou
            })
            
    # Aggregate results
    mean_dice = np.mean([m['dice'] for m in all_metrics])
    mean_iou = np.mean([m['iou'] for m in all_metrics])
    
    print(f"\nResults for {args.checkpoint}:")
    print(f"Mean Dice: {mean_dice:.4f}")
    print(f"Mean IoU: {mean_iou:.4f}")
    
    # Save results
    output_file = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(output_file, 'w') as f:
        json.dump({
            'checkpoint': args.checkpoint,
            'mean_dice': mean_dice,
            'mean_iou': mean_iou,
            'per_image': all_metrics
        }, f, indent=2)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--config', type=str, default='sam2.1_hiera_l.yaml', help='Model config')
    parser.add_argument('--data_dir', type=str, default='data/bcss', help='Data directory')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    evaluate_checkpoint(args)
