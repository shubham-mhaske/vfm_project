#!/usr/bin/env python3
"""
Comprehensive Metrics Collection Script

This script re-runs all experiments to collect complete metrics including:
- Segmentation: Dice, IoU, Std Dev (per-class and overall)
- Classification: Accuracy, Precision, Recall, F1-Score (per-class and overall)

Run on HPRC with:
    sbatch scripts/slurm/run_complete_metrics.slurm

Or locally:
    python scripts/collect_all_metrics.py --local

Author: VFM Project
Date: December 2025
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, os.path.join(project_root, 'sam2'))

import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

# Local imports
from dataset import BCSSDataset
from sam_segmentation import (
    get_sam2_predictor, get_prompts_from_mask, 
    get_predicted_mask_from_prompts, postprocess_mask, calculate_metrics
)
from clip_classification import CLIPClassifier, crop_region_from_mask, load_prompts_from_json
from device_utils import get_device


def log(msg):
    """Print with timestamp."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")
    sys.stdout.flush()


class ComprehensiveMetricsCollector:
    """Collects complete segmentation and classification metrics."""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.project_root = project_root
        
        # Paths
        self.image_dir = os.path.join(project_root, 'data', 'bcss', 'images')
        self.mask_dir = os.path.join(project_root, 'data', 'bcss', 'masks')
        
        # Class mappings
        self.class_names = {
            0: 'background',
            1: 'tumor',
            2: 'stroma', 
            3: 'lymphocyte',
            4: 'necrosis',
            18: 'blood_vessel'
        }
        self.target_class_ids = [1, 2, 3, 4, 18]
        self.class_name_list = ['tumor', 'stroma', 'lymphocyte', 'necrosis', 'blood_vessel']
        
        # Results storage
        self.results = {}
        
    def load_dataset(self, split='test'):
        """Load BCSS dataset."""
        log(f"Loading BCSS {split} dataset...")
        dataset = BCSSDataset(
            image_dir=self.image_dir,
            mask_dir=self.mask_dir,
            split=split
        )
        log(f"  Loaded {len(dataset)} samples")
        return dataset
    
    def evaluate_sam2_segmentation(self, prompt_configs, sam_config, sam_checkpoint, 
                                    split='test', max_samples=None):
        """
        Evaluate SAM2 segmentation with multiple prompt configurations.
        
        Args:
            prompt_configs: List of dicts with 'name', 'prompt_type', 'use_neg_points', 'use_tta'
            sam_config: Path to SAM2 config
            sam_checkpoint: Path to SAM2 checkpoint
            split: Dataset split
            max_samples: Limit samples for debugging
            
        Returns:
            Dict with results for each configuration
        """
        log("\n" + "="*70)
        log("SAM2 SEGMENTATION EVALUATION")
        log("="*70)
        
        # Load dataset
        dataset = self.load_dataset(split)
        num_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))
        
        # Load SAM2 predictor
        log(f"\nLoading SAM2 predictor...")
        log(f"  Config: {sam_config}")
        log(f"  Checkpoint: {sam_checkpoint}")
        
        # Change to sam2 directory for config resolution
        orig_cwd = os.getcwd()
        os.chdir(os.path.join(project_root, 'sam2'))
        
        try:
            # Register OmegaConf resolvers if needed
            try:
                from training.utils.train_utils import register_omegaconf_resolvers
                register_omegaconf_resolvers()
            except:
                pass
            
            predictor = get_sam2_predictor(
                sam_config,
                os.path.join(project_root, sam_checkpoint),
                self.device
            )
        finally:
            os.chdir(orig_cwd)
        
        log(f"  SAM2 loaded on {self.device}")
        
        # Results for each configuration
        all_results = {}
        
        for config in prompt_configs:
            config_name = config['name']
            log(f"\n--- Evaluating: {config_name} ---")
            
            # Per-class metrics storage
            class_metrics = {cid: {'dice': [], 'iou': []} for cid in self.target_class_ids}
            sample_results = []
            
            for i in tqdm(range(num_samples), desc=config_name, disable=False):
                sample = dataset[i]
                image = sample['image_np']
                gt_mask = sample['mask'].numpy()
                unique_classes = np.unique(gt_mask)
                
                predictor.set_image(image)
                
                for class_id in unique_classes:
                    if class_id == 0 or class_id not in self.class_names:
                        continue
                    
                    class_name = self.class_names[class_id]
                    binary_gt = (gt_mask == class_id).astype(np.uint8)
                    
                    if binary_gt.sum() == 0:
                        continue
                    
                    prompts = get_prompts_from_mask(binary_gt)
                    prompt_type = config['prompt_type']
                    
                    if prompt_type not in prompts:
                        continue
                    
                    # Get prediction
                    if config.get('use_tta', False):
                        from tta_utils import predict_with_tta
                        pred_mask = predict_with_tta(
                            predictor, image, prompts,
                            prompt_type=prompt_type,
                            use_neg_points=config.get('use_neg_points', False)
                        )
                    else:
                        pred_mask, _, _ = get_predicted_mask_from_prompts(
                            predictor, image, prompts,
                            prompt_type=prompt_type,
                            use_neg_points=config.get('use_neg_points', False)
                        )
                        pred_mask = (pred_mask > 0.5).astype(np.uint8)
                    
                    # Calculate metrics
                    dice, iou = calculate_metrics(pred_mask, binary_gt)
                    
                    class_metrics[class_id]['dice'].append(dice)
                    class_metrics[class_id]['iou'].append(iou)
                    
                    sample_results.append({
                        'sample_idx': i,
                        'class_id': int(class_id),
                        'class_name': class_name,
                        'dice': float(dice),
                        'iou': float(iou),
                        'filename': dataset.image_files[i]
                    })
            
            # Aggregate results
            config_results = {
                'config': config,
                'per_class': {},
                'overall': {},
                'sample_results': sample_results
            }
            
            all_dice = []
            all_iou = []
            
            for cid in self.target_class_ids:
                cname = self.class_names[cid]
                dice_scores = class_metrics[cid]['dice']
                iou_scores = class_metrics[cid]['iou']
                
                if dice_scores:
                    config_results['per_class'][cname] = {
                        'dice_mean': float(np.mean(dice_scores)),
                        'dice_std': float(np.std(dice_scores)),
                        'iou_mean': float(np.mean(iou_scores)),
                        'iou_std': float(np.std(iou_scores)),
                        'count': len(dice_scores)
                    }
                    all_dice.extend(dice_scores)
                    all_iou.extend(iou_scores)
                else:
                    config_results['per_class'][cname] = {
                        'dice_mean': 0, 'dice_std': 0,
                        'iou_mean': 0, 'iou_std': 0,
                        'count': 0
                    }
            
            config_results['overall'] = {
                'dice_mean': float(np.mean(all_dice)) if all_dice else 0,
                'dice_std': float(np.std(all_dice)) if all_dice else 0,
                'iou_mean': float(np.mean(all_iou)) if all_iou else 0,
                'iou_std': float(np.std(all_iou)) if all_iou else 0,
                'total_samples': len(all_dice)
            }
            
            all_results[config_name] = config_results
            
            # Print summary
            log(f"\n  Results for {config_name}:")
            log(f"    Overall Dice: {config_results['overall']['dice_mean']:.4f} ± {config_results['overall']['dice_std']:.4f}")
            log(f"    Overall IoU:  {config_results['overall']['iou_mean']:.4f} ± {config_results['overall']['iou_std']:.4f}")
        
        return all_results
    
    def evaluate_clip_classification(self, prompt_files, split='test', max_samples=None):
        """
        Evaluate CLIP classification with multiple prompt configurations.
        
        Args:
            prompt_files: List of dicts with 'name', 'path'
            split: Dataset split
            max_samples: Limit samples for debugging
            
        Returns:
            Dict with results for each prompt configuration
        """
        log("\n" + "="*70)
        log("CLIP CLASSIFICATION EVALUATION")
        log("="*70)
        
        # Load dataset
        dataset = self.load_dataset(split)
        num_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))
        
        # Load CLIP classifier
        log("\nLoading CLIP classifier...")
        clip_classifier = CLIPClassifier(device=self.device)
        log("  CLIP loaded!")
        
        all_results = {}
        
        for prompt_config in prompt_files:
            prompt_name = prompt_config['name']
            prompt_path = os.path.join(project_root, prompt_config['path'])
            
            log(f"\n--- Evaluating: {prompt_name} ---")
            log(f"    Prompt file: {prompt_path}")
            
            if not os.path.exists(prompt_path):
                log(f"    WARNING: Prompt file not found, skipping...")
                continue
            
            prompts = load_prompts_from_json(prompt_path)
            
            # Collect predictions
            y_true = []
            y_pred = []
            per_class_correct = {cname: 0 for cname in self.class_name_list}
            per_class_total = {cname: 0 for cname in self.class_name_list}
            
            for i in tqdm(range(num_samples), desc=prompt_name, disable=False):
                sample = dataset[i]
                image = sample['image_np']
                gt_mask = sample['mask'].numpy()
                unique_classes = np.unique(gt_mask)
                
                for class_id in unique_classes:
                    if class_id == 0 or class_id not in self.class_names:
                        continue
                    
                    gt_class_name = self.class_names[class_id]
                    binary_mask = (gt_mask == class_id).astype(np.uint8)
                    
                    if binary_mask.sum() == 0:
                        continue
                    
                    # Crop region for classification
                    cropped = crop_region_from_mask(image, binary_mask)
                    if cropped is None:
                        continue
                    
                    # Classify
                    pred_class = clip_classifier.classify_region(cropped, prompts)
                    if pred_class is None:
                        continue
                    
                    y_true.append(gt_class_name)
                    y_pred.append(pred_class)
                    
                    per_class_total[gt_class_name] += 1
                    if pred_class == gt_class_name:
                        per_class_correct[gt_class_name] += 1
            
            # Calculate metrics
            if len(y_true) == 0:
                log(f"    No valid predictions, skipping...")
                continue
            
            # Overall accuracy
            accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
            
            # Per-class metrics
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true, y_pred, labels=self.class_name_list, zero_division=0
            )
            
            # Build results
            config_results = {
                'prompt_file': prompt_config['path'],
                'overall': {
                    'accuracy': float(accuracy),
                    'total_samples': len(y_true)
                },
                'per_class': {}
            }
            
            for idx, cname in enumerate(self.class_name_list):
                config_results['per_class'][cname] = {
                    'precision': float(precision[idx]),
                    'recall': float(recall[idx]),
                    'f1_score': float(f1[idx]),
                    'support': int(support[idx]),
                    'correct': per_class_correct[cname],
                    'total': per_class_total[cname],
                    'accuracy': float(per_class_correct[cname] / per_class_total[cname]) if per_class_total[cname] > 0 else 0
                }
            
            # Macro and weighted averages
            config_results['macro_avg'] = {
                'precision': float(np.mean(precision)),
                'recall': float(np.mean(recall)),
                'f1_score': float(np.mean(f1))
            }
            
            weights = support / support.sum() if support.sum() > 0 else np.zeros_like(support)
            config_results['weighted_avg'] = {
                'precision': float(np.average(precision, weights=weights)) if weights.sum() > 0 else 0,
                'recall': float(np.average(recall, weights=weights)) if weights.sum() > 0 else 0,
                'f1_score': float(np.average(f1, weights=weights)) if weights.sum() > 0 else 0
            }
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred, labels=self.class_name_list)
            config_results['confusion_matrix'] = cm.tolist()
            
            all_results[prompt_name] = config_results
            
            # Print summary
            log(f"\n  Results for {prompt_name}:")
            log(f"    Accuracy: {accuracy:.2%}")
            log(f"    Macro F1: {config_results['macro_avg']['f1_score']:.4f}")
        
        return all_results
    
    def evaluate_medsam(self, checkpoint_path, split='test', use_tta=True, max_samples=None):
        """Evaluate MedSAM segmentation."""
        log("\n" + "="*70)
        log("MEDSAM SEGMENTATION EVALUATION")
        log("="*70)
        
        # Import MedSAM evaluation functions
        sys.path.insert(0, os.path.join(project_root, 'MedSAM'))
        from evaluate_medsam import (
            load_medsam_model, preprocess_image_for_medsam,
            medsam_inference, predict_with_medsam_tta,
            scale_box_to_1024
        )
        
        # Load dataset
        dataset = self.load_dataset(split)
        num_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))
        
        # Load MedSAM
        full_ckpt_path = os.path.join(project_root, checkpoint_path)
        log(f"\nLoading MedSAM from: {full_ckpt_path}")
        medsam_model = load_medsam_model(full_ckpt_path, self.device)
        
        # Evaluate
        class_metrics = {cid: {'dice': [], 'iou': []} for cid in self.target_class_ids}
        
        for i in tqdm(range(num_samples), desc="MedSAM", disable=False):
            sample = dataset[i]
            image = sample['image_np']
            gt_mask = sample['mask'].numpy()
            unique_classes = np.unique(gt_mask)
            
            H, W = image.shape[:2]
            
            for class_id in unique_classes:
                if class_id == 0 or class_id not in self.class_names:
                    continue
                
                binary_gt = (gt_mask == class_id).astype(np.uint8)
                if binary_gt.sum() == 0:
                    continue
                
                prompts = get_prompts_from_mask(binary_gt)
                if 'box' not in prompts:
                    continue
                
                box = prompts['box']
                
                if use_tta:
                    pred_mask, _ = predict_with_medsam_tta(medsam_model, image, box, self.device)
                else:
                    # Use cached embedding for efficiency
                    from evaluate_medsam import preprocess_image_for_medsam
                    img_1024, _, H, W = preprocess_image_for_medsam(image)
                    img_tensor = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        embedding = medsam_model.image_encoder(img_tensor)
                    
                    box_1024 = scale_box_to_1024(box, H, W)
                    box_1024 = np.array([box_1024])
                    pred_mask, _ = medsam_inference(medsam_model, embedding, box_1024, H, W, self.device)
                
                dice, iou = calculate_metrics(pred_mask, binary_gt)
                class_metrics[class_id]['dice'].append(dice)
                class_metrics[class_id]['iou'].append(iou)
        
        # Aggregate
        results = {'per_class': {}, 'overall': {}}
        all_dice, all_iou = [], []
        
        for cid in self.target_class_ids:
            cname = self.class_names[cid]
            dice_scores = class_metrics[cid]['dice']
            iou_scores = class_metrics[cid]['iou']
            
            if dice_scores:
                results['per_class'][cname] = {
                    'dice_mean': float(np.mean(dice_scores)),
                    'dice_std': float(np.std(dice_scores)),
                    'iou_mean': float(np.mean(iou_scores)),
                    'iou_std': float(np.std(iou_scores)),
                    'count': len(dice_scores)
                }
                all_dice.extend(dice_scores)
                all_iou.extend(iou_scores)
        
        results['overall'] = {
            'dice_mean': float(np.mean(all_dice)) if all_dice else 0,
            'dice_std': float(np.std(all_dice)) if all_dice else 0,
            'iou_mean': float(np.mean(all_iou)) if all_iou else 0,
            'iou_std': float(np.std(all_iou)) if all_iou else 0,
            'total_samples': len(all_dice)
        }
        
        results['_metadata'] = {
            'model': 'MedSAM',
            'checkpoint': checkpoint_path,
            'use_tta': use_tta,
            'split': split,
            'timestamp': datetime.now().isoformat()
        }
        
        log(f"\n  MedSAM Results (TTA={use_tta}):")
        log(f"    Dice: {results['overall']['dice_mean']:.4f} ± {results['overall']['dice_std']:.4f}")
        log(f"    IoU:  {results['overall']['iou_mean']:.4f} ± {results['overall']['iou_std']:.4f}")
        
        return results
    
    def run_all_experiments(self, max_samples=None, skip_medsam=False):
        """Run all experiments and collect comprehensive metrics."""
        
        output_dir = os.path.join(project_root, 'results', 'complete_metrics')
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # ==============================
        # 1. SAM2 Segmentation Experiments
        # ==============================
        sam_prompt_configs = [
            {'name': 'centroid', 'prompt_type': 'centroid', 'use_neg_points': False, 'use_tta': False},
            {'name': 'multi_point', 'prompt_type': 'multi_point', 'use_neg_points': False, 'use_tta': False},
            {'name': 'box_baseline', 'prompt_type': 'box', 'use_neg_points': False, 'use_tta': False},
            {'name': 'box_neg_points', 'prompt_type': 'box', 'use_neg_points': True, 'use_tta': False},
            {'name': 'box_tta', 'prompt_type': 'box', 'use_neg_points': False, 'use_tta': True},
            {'name': 'box_neg_tta', 'prompt_type': 'box', 'use_neg_points': True, 'use_tta': True},
        ]
        
        sam_results = self.evaluate_sam2_segmentation(
            prompt_configs=sam_prompt_configs,
            sam_config='configs/sam2.1/sam2.1_hiera_l.yaml',
            sam_checkpoint='sam2/checkpoints/sam2.1_hiera_large.pt',
            split='test',
            max_samples=max_samples
        )
        
        # Save SAM2 results
        sam_output_path = os.path.join(output_dir, f'sam2_segmentation_{timestamp}.json')
        with open(sam_output_path, 'w') as f:
            # Remove sample_results for cleaner output (can be very large)
            sam_clean = {}
            for name, data in sam_results.items():
                sam_clean[name] = {k: v for k, v in data.items() if k != 'sample_results'}
            json.dump(sam_clean, f, indent=2)
        log(f"\nSAM2 results saved to: {sam_output_path}")
        
        # ==============================
        # 2. CLIP Classification Experiments  
        # ==============================
        clip_prompt_configs = [
            {'name': 'hardcoded_v1', 'path': 'configs/prompts/hard_coded_prompts.json'},
            {'name': 'hardcoded_v2', 'path': 'configs/prompts/hard_coded_prompts_v2.json'},
            {'name': 'llm_text_v1_jargon', 'path': 'configs/prompts/llm_text_prompts_v1_gemini_pro_latest.json'},
            {'name': 'llm_text_v2_clip_friendly', 'path': 'configs/prompts/llm_text_prompts_v2_clip_friendly.json'},
            {'name': 'llm_text_v3_fewshot', 'path': 'configs/prompts/llm_text_prompts_v3_fewshot.json'},
            {'name': 'llm_multimodal_v1', 'path': 'configs/prompts/llm_multimodal_prompts_v1_gemini_2.5_flash.json'},
            {'name': 'llm_multimodal_v2_clip_friendly', 'path': 'configs/prompts/llm_multimodal_prompts_v2_clip_friendly.json'},
            {'name': 'llm_multimodal_v3_fewshot', 'path': 'configs/prompts/llm_multimodal_prompts_v3_fewshot.json'},
        ]
        
        clip_results = self.evaluate_clip_classification(
            prompt_files=clip_prompt_configs,
            split='test',
            max_samples=max_samples
        )
        
        # Save CLIP results
        clip_output_path = os.path.join(output_dir, f'clip_classification_{timestamp}.json')
        with open(clip_output_path, 'w') as f:
            json.dump(clip_results, f, indent=2)
        log(f"\nCLIP results saved to: {clip_output_path}")
        
        # ==============================
        # 3. MedSAM Experiments
        # ==============================
        if not skip_medsam:
            medsam_ckpt = 'models/medsam_checkpoints/medsam_vit_b.pth'
            if os.path.exists(os.path.join(project_root, medsam_ckpt)):
                # Without TTA
                medsam_results_no_tta = self.evaluate_medsam(
                    checkpoint_path=medsam_ckpt,
                    split='test',
                    use_tta=False,
                    max_samples=max_samples
                )
                
                # With TTA
                medsam_results_tta = self.evaluate_medsam(
                    checkpoint_path=medsam_ckpt,
                    split='test', 
                    use_tta=True,
                    max_samples=max_samples
                )
                
                medsam_combined = {
                    'medsam_box': medsam_results_no_tta,
                    'medsam_box_tta': medsam_results_tta
                }
                
                medsam_output_path = os.path.join(output_dir, f'medsam_segmentation_{timestamp}.json')
                with open(medsam_output_path, 'w') as f:
                    json.dump(medsam_combined, f, indent=2)
                log(f"\nMedSAM results saved to: {medsam_output_path}")
            else:
                log(f"\nWARNING: MedSAM checkpoint not found at {medsam_ckpt}, skipping...")
        
        # ==============================
        # 4. Generate Summary Report
        # ==============================
        self.generate_summary_report(sam_results, clip_results, output_dir, timestamp)
        
        log("\n" + "="*70)
        log("ALL EXPERIMENTS COMPLETE!")
        log(f"Results saved to: {output_dir}")
        log("="*70)
        
        return {
            'sam2': sam_results,
            'clip': clip_results
        }
    
    def generate_summary_report(self, sam_results, clip_results, output_dir, timestamp):
        """Generate a comprehensive summary report."""
        
        report = {
            'timestamp': timestamp,
            'summary': {
                'segmentation': {},
                'classification': {}
            },
            'detailed': {
                'segmentation': sam_results,
                'classification': clip_results
            }
        }
        
        # Best segmentation
        best_seg_name = None
        best_seg_dice = 0
        for name, data in sam_results.items():
            dice = data['overall']['dice_mean']
            if dice > best_seg_dice:
                best_seg_dice = dice
                best_seg_name = name
        
        report['summary']['segmentation'] = {
            'best_config': best_seg_name,
            'best_dice': best_seg_dice,
            'best_iou': sam_results[best_seg_name]['overall']['iou_mean'] if best_seg_name else 0,
            'all_configs': {name: data['overall']['dice_mean'] for name, data in sam_results.items()}
        }
        
        # Best classification
        best_clf_name = None
        best_clf_acc = 0
        for name, data in clip_results.items():
            acc = data['overall']['accuracy']
            if acc > best_clf_acc:
                best_clf_acc = acc
                best_clf_name = name
        
        report['summary']['classification'] = {
            'best_config': best_clf_name,
            'best_accuracy': best_clf_acc,
            'best_macro_f1': clip_results[best_clf_name]['macro_avg']['f1_score'] if best_clf_name else 0,
            'all_configs': {name: data['overall']['accuracy'] for name, data in clip_results.items()}
        }
        
        # Save summary
        summary_path = os.path.join(output_dir, f'summary_report_{timestamp}.json')
        with open(summary_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        log("\n" + "="*70)
        log("SUMMARY REPORT")
        log("="*70)
        
        log("\nSegmentation Results (Dice | IoU):")
        log("-"*50)
        for name, data in sorted(sam_results.items(), key=lambda x: x[1]['overall']['dice_mean'], reverse=True):
            dice = data['overall']['dice_mean']
            iou = data['overall']['iou_mean']
            marker = " 🏆" if name == best_seg_name else ""
            log(f"  {name:20s}: {dice:.4f} | {iou:.4f}{marker}")
        
        log("\nClassification Results (Accuracy | Macro-F1):")
        log("-"*50)
        for name, data in sorted(clip_results.items(), key=lambda x: x[1]['overall']['accuracy'], reverse=True):
            acc = data['overall']['accuracy']
            f1 = data['macro_avg']['f1_score']
            marker = " 🏆" if name == best_clf_name else ""
            log(f"  {name:30s}: {acc:.2%} | {f1:.4f}{marker}")
        
        log(f"\nSummary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect comprehensive metrics for all experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--local', action='store_true',
                        help='Run locally (use CPU if no GPU)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Limit samples for debugging')
    parser.add_argument('--skip_medsam', action='store_true',
                        help='Skip MedSAM evaluation')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID to use')
    
    args = parser.parse_args()
    
    # Device setup
    if args.local and not torch.cuda.is_available():
        device = 'cpu'
    else:
        device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    
    log(f"Using device: {device}")
    if 'cuda' in device:
        log(f"GPU: {torch.cuda.get_device_name(args.gpu)}")
    
    # Run experiments
    collector = ComprehensiveMetricsCollector(device=device)
    collector.run_all_experiments(
        max_samples=args.max_samples,
        skip_medsam=args.skip_medsam
    )


if __name__ == '__main__':
    main()
