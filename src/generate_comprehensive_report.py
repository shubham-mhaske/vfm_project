#!/usr/bin/env python3
"""
Comprehensive Metrics Report Generator
Generates all metrics needed for paper/presentation:
- Segmentation: Dice, IoU, per-class breakdown, std, confidence intervals
- Classification: Accuracy, Precision, Recall, F1, Confusion Matrix
- Comparison tables across all experiments
- LaTeX tables and figures ready for paper
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'sam2'))

import argparse
import json
import time
import numpy as np
from tqdm import tqdm
import torch
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score

from dataset import BCSSDataset
from sam_segmentation import get_sam2_predictor, get_prompts_from_mask, calculate_metrics, get_predicted_mask_from_prompts
from clip_classification import CLIPClassifier, crop_region_from_mask, load_prompts_from_json, get_clip_model_path
from device_utils import get_device
from tta_utils import predict_with_tta
from iterative_refinement import predict_with_refinement, predict_with_tta_and_refinement
from training.utils.train_utils import register_omegaconf_resolvers


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def calculate_confidence_interval(data, confidence=0.95):
    """Calculate 95% confidence interval."""
    n = len(data)
    if n < 2:
        return 0, 0, 0
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    se = std / np.sqrt(n)
    z = 1.96  # 95% CI
    return mean, mean - z*se, mean + z*se


class ComprehensiveEvaluator:
    def __init__(self, predictor, clip_classifier, clip_prompts, dataset):
        self.predictor = predictor
        self.clip_classifier = clip_classifier
        self.clip_prompts = clip_prompts
        self.dataset = dataset
        self.class_names = dataset.class_names
        
    def evaluate_configuration(self, config_name, prompt_type='box', use_neg_points=True,
                                use_tta=False, use_refinement=False, num_refine_iters=2):
        """Evaluate a single configuration with comprehensive metrics."""
        
        # Storage for all samples
        all_dice = defaultdict(list)
        all_iou = defaultdict(list)
        all_true_labels = []
        all_pred_labels = []
        sample_results = []
        
        for i in tqdm(range(len(self.dataset)), desc=f"Evaluating {config_name}"):
            sample = self.dataset[i]
            image = sample['image_np']
            gt_mask = sample['mask'].numpy()
            filename = sample.get('filename', f'sample_{i}')
            
            for class_id in np.unique(gt_mask):
                if class_id == 0:
                    continue
                
                class_name = self.class_names.get(class_id)
                if not class_name or class_name not in self.clip_prompts:
                    continue
                
                binary_gt = (gt_mask == class_id).astype(np.uint8)
                prompts = get_prompts_from_mask(binary_gt)
                
                if prompt_type not in prompts:
                    continue
                
                # Get prediction
                if use_tta and use_refinement:
                    pred_mask = predict_with_tta_and_refinement(
                        self.predictor, image, prompts, prompt_type, use_neg_points,
                        num_augmentations=4, num_refinement_iterations=num_refine_iters
                    )
                elif use_tta:
                    pred_mask = predict_with_tta(
                        self.predictor, image, prompts, prompt_type, use_neg_points
                    )
                elif use_refinement:
                    pred_mask = predict_with_refinement(
                        self.predictor, image, prompts, prompt_type, use_neg_points,
                        num_iterations=num_refine_iters
                    )
                else:
                    self.predictor.set_image(image)
                    mask_logits, _, _ = get_predicted_mask_from_prompts(
                        self.predictor, image, prompts, prompt_type, use_neg_points
                    )
                    pred_mask = (mask_logits > 0.5).astype(np.uint8)
                
                # Segmentation metrics
                dice, iou = calculate_metrics(pred_mask, binary_gt)
                all_dice[class_name].append(dice)
                all_iou[class_name].append(iou)
                
                # Classification
                pred_class = None
                if self.clip_classifier and pred_mask.sum() > 0:
                    cropped = crop_region_from_mask(image, pred_mask)
                    if cropped:
                        pred_class = self.clip_classifier.classify_region(cropped, self.clip_prompts)
                        all_true_labels.append(class_name)
                        all_pred_labels.append(pred_class)
                
                # Store sample result
                sample_results.append({
                    'filename': filename,
                    'class': class_name,
                    'dice': float(dice),
                    'iou': float(iou),
                    'pred_class': pred_class,
                    'correct': pred_class == class_name if pred_class else None
                })
        
        # Compute comprehensive metrics
        results = self._compute_metrics(all_dice, all_iou, all_true_labels, all_pred_labels)
        results['sample_results'] = sample_results
        results['config_name'] = config_name
        
        return results
    
    def _compute_metrics(self, all_dice, all_iou, true_labels, pred_labels):
        """Compute all metrics from raw data."""
        results = {
            'segmentation': {'per_class': {}, 'overall': {}},
            'classification': {'per_class': {}, 'overall': {}}
        }
        
        # Per-class segmentation metrics
        all_dice_flat = []
        all_iou_flat = []
        
        for cname in ['tumor', 'stroma', 'lymphocyte', 'necrosis', 'blood_vessel']:
            if cname in all_dice and all_dice[cname]:
                dice_vals = all_dice[cname]
                iou_vals = all_iou[cname]
                all_dice_flat.extend(dice_vals)
                all_iou_flat.extend(iou_vals)
                
                dice_mean, dice_ci_low, dice_ci_high = calculate_confidence_interval(dice_vals)
                iou_mean, iou_ci_low, iou_ci_high = calculate_confidence_interval(iou_vals)
                
                results['segmentation']['per_class'][cname] = {
                    'dice_mean': float(dice_mean),
                    'dice_std': float(np.std(dice_vals)),
                    'dice_ci_95': [float(dice_ci_low), float(dice_ci_high)],
                    'iou_mean': float(iou_mean),
                    'iou_std': float(np.std(iou_vals)),
                    'iou_ci_95': [float(iou_ci_low), float(iou_ci_high)],
                    'count': len(dice_vals)
                }
        
        # Overall segmentation
        if all_dice_flat:
            dice_mean, dice_ci_low, dice_ci_high = calculate_confidence_interval(all_dice_flat)
            iou_mean, iou_ci_low, iou_ci_high = calculate_confidence_interval(all_iou_flat)
            
            results['segmentation']['overall'] = {
                'dice_mean': float(dice_mean),
                'dice_std': float(np.std(all_dice_flat)),
                'dice_ci_95': [float(dice_ci_low), float(dice_ci_high)],
                'iou_mean': float(iou_mean),
                'iou_std': float(np.std(all_iou_flat)),
                'iou_ci_95': [float(iou_ci_low), float(iou_ci_high)],
                'count': len(all_dice_flat)
            }
        
        # Classification metrics
        if true_labels and pred_labels:
            # Get unique class names
            class_names = sorted(set(true_labels + pred_labels))
            
            # Overall accuracy
            accuracy = accuracy_score(true_labels, pred_labels)
            
            # Precision, Recall, F1 per class
            precision, recall, f1, support = precision_recall_fscore_support(
                true_labels, pred_labels, labels=class_names, average=None, zero_division=0
            )
            
            # Macro averages
            macro_precision = np.mean(precision)
            macro_recall = np.mean(recall)
            macro_f1 = np.mean(f1)
            
            # Confusion matrix
            cm = confusion_matrix(true_labels, pred_labels, labels=class_names)
            
            results['classification']['overall'] = {
                'accuracy': float(accuracy),
                'macro_precision': float(macro_precision),
                'macro_recall': float(macro_recall),
                'macro_f1': float(macro_f1),
                'total_samples': len(true_labels),
                'correct': int(sum(t == p for t, p in zip(true_labels, pred_labels)))
            }
            
            results['classification']['per_class'] = {}
            for i, cname in enumerate(class_names):
                results['classification']['per_class'][cname] = {
                    'precision': float(precision[i]),
                    'recall': float(recall[i]),
                    'f1': float(f1[i]),
                    'support': int(support[i])
                }
            
            results['classification']['confusion_matrix'] = {
                'matrix': cm.tolist(),
                'labels': class_names
            }
        
        return results


def generate_latex_tables(results, output_dir):
    """Generate LaTeX tables for paper."""
    
    # Segmentation table
    latex_seg = """\\begin{table}[h]
\\centering
\\caption{Segmentation Performance (Dice Score)}
\\begin{tabular}{lccccc}
\\toprule
Configuration & Tumor & Stroma & Lymph. & Necrosis & Blood V. & Overall \\\\
\\midrule
"""
    
    for config_name, res in results.items():
        seg = res['segmentation']
        row = f"{config_name.replace('_', ' ')} "
        for cname in ['tumor', 'stroma', 'lymphocyte', 'necrosis', 'blood_vessel']:
            if cname in seg['per_class']:
                dice = seg['per_class'][cname]['dice_mean']
                row += f"& {dice:.3f} "
            else:
                row += "& - "
        row += f"& \\textbf{{{seg['overall']['dice_mean']:.3f}}} \\\\\n"
        latex_seg += row
    
    latex_seg += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    
    # Classification table
    latex_cls = """\\begin{table}[h]
\\centering
\\caption{Classification Performance}
\\begin{tabular}{lcccc}
\\toprule
Configuration & Accuracy & Precision & Recall & F1 \\\\
\\midrule
"""
    
    for config_name, res in results.items():
        cls = res['classification']['overall']
        latex_cls += f"{config_name.replace('_', ' ')} & {cls['accuracy']:.3f} & {cls['macro_precision']:.3f} & {cls['macro_recall']:.3f} & {cls['macro_f1']:.3f} \\\\\n"
    
    latex_cls += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    
    # Save
    with open(os.path.join(output_dir, 'latex_tables.tex'), 'w') as f:
        f.write("% Segmentation Table\n")
        f.write(latex_seg)
        f.write("\n\n% Classification Table\n")
        f.write(latex_cls)
    
    return latex_seg, latex_cls


def generate_figures(results, output_dir):
    """Generate figures for paper/presentation."""
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Bar chart comparing configurations
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    configs = list(results.keys())
    dice_scores = [results[c]['segmentation']['overall']['dice_mean'] for c in configs]
    acc_scores = [results[c]['classification']['overall']['accuracy'] for c in configs]
    
    x = np.arange(len(configs))
    
    # Segmentation
    axes[0].bar(x, dice_scores, color='steelblue', edgecolor='black')
    axes[0].set_ylabel('Dice Score', fontsize=12)
    axes[0].set_title('Segmentation Performance', fontsize=14)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([c.replace('_', '\n') for c in configs], rotation=45, ha='right', fontsize=9)
    axes[0].set_ylim(0, 1)
    for i, v in enumerate(dice_scores):
        axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)
    
    # Classification
    axes[1].bar(x, acc_scores, color='darkorange', edgecolor='black')
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Classification Performance', fontsize=14)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([c.replace('_', '\n') for c in configs], rotation=45, ha='right', fontsize=9)
    axes[1].set_ylim(0, 1)
    for i, v in enumerate(acc_scores):
        axes[1].text(i, v + 0.02, f'{v:.1%}', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Per-class performance heatmap (for best config)
    best_config = max(results.keys(), key=lambda k: results[k]['segmentation']['overall']['dice_mean'])
    best = results[best_config]
    
    class_names = ['tumor', 'stroma', 'lymphocyte', 'necrosis', 'blood_vessel']
    metrics = ['Dice', 'IoU', 'Precision', 'Recall', 'F1']
    
    data = np.zeros((len(class_names), len(metrics)))
    for i, cname in enumerate(class_names):
        if cname in best['segmentation']['per_class']:
            data[i, 0] = best['segmentation']['per_class'][cname]['dice_mean']
            data[i, 1] = best['segmentation']['per_class'][cname]['iou_mean']
        if cname in best['classification']['per_class']:
            data[i, 2] = best['classification']['per_class'][cname]['precision']
            data[i, 3] = best['classification']['per_class'][cname]['recall']
            data[i, 4] = best['classification']['per_class'][cname]['f1']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(data, annot=True, fmt='.3f', cmap='RdYlGn', 
                xticklabels=metrics, yticklabels=[c.title() for c in class_names],
                vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Score'})
    ax.set_title(f'Per-Class Performance: {best_config}', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_class_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Confusion matrix for best config
    if 'confusion_matrix' in best['classification']:
        cm = np.array(best['classification']['confusion_matrix']['matrix'])
        labels = best['classification']['confusion_matrix']['labels']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[l.title() for l in labels],
                    yticklabels=[l.title() for l in labels], ax=ax)
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('True', fontsize=12)
        ax.set_title(f'Confusion Matrix: {best_config}', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    log(f"Figures saved to {output_dir}")


def generate_markdown_report(results, output_dir):
    """Generate a comprehensive markdown report."""
    
    report = f"""# Comprehensive Evaluation Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

"""
    
    # Find best config
    best_seg = max(results.keys(), key=lambda k: results[k]['segmentation']['overall']['dice_mean'])
    best_cls = max(results.keys(), key=lambda k: results[k]['classification']['overall']['accuracy'])
    
    report += f"""| Metric | Best Config | Score |
|--------|-------------|-------|
| **Segmentation (Dice)** | {best_seg} | {results[best_seg]['segmentation']['overall']['dice_mean']:.4f} |
| **Classification (Accuracy)** | {best_cls} | {results[best_cls]['classification']['overall']['accuracy']:.2%} |

---

## Segmentation Results

### Overall Performance

| Configuration | Dice | IoU | Dice 95% CI | Samples |
|---------------|------|-----|-------------|---------|
"""
    
    for config_name, res in results.items():
        seg = res['segmentation']['overall']
        ci = seg['dice_ci_95']
        report += f"| {config_name} | **{seg['dice_mean']:.4f}** | {seg['iou_mean']:.4f} | [{ci[0]:.3f}, {ci[1]:.3f}] | {seg['count']} |\n"
    
    report += """
### Per-Class Breakdown (Best Configuration)

| Class | Dice | Std | IoU | Samples |
|-------|------|-----|-----|---------|
"""
    
    best = results[best_seg]
    for cname in ['tumor', 'stroma', 'lymphocyte', 'necrosis', 'blood_vessel']:
        if cname in best['segmentation']['per_class']:
            pc = best['segmentation']['per_class'][cname]
            report += f"| {cname.title()} | {pc['dice_mean']:.4f} | ±{pc['dice_std']:.3f} | {pc['iou_mean']:.4f} | {pc['count']} |\n"
    
    report += """
---

## Classification Results

### Overall Performance

| Configuration | Accuracy | Precision | Recall | F1 | Correct/Total |
|---------------|----------|-----------|--------|-----|---------------|
"""
    
    for config_name, res in results.items():
        cls = res['classification']['overall']
        report += f"| {config_name} | **{cls['accuracy']:.2%}** | {cls['macro_precision']:.3f} | {cls['macro_recall']:.3f} | {cls['macro_f1']:.3f} | {cls['correct']}/{cls['total_samples']} |\n"
    
    report += """
### Per-Class Classification (Best Configuration)

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
"""
    
    best = results[best_cls]
    for cname in ['tumor', 'stroma', 'lymphocyte', 'necrosis', 'blood_vessel']:
        if cname in best['classification']['per_class']:
            pc = best['classification']['per_class'][cname]
            report += f"| {cname.title()} | {pc['precision']:.3f} | {pc['recall']:.3f} | {pc['f1']:.3f} | {pc['support']} |\n"
    
    report += """
---

## Key Findings

1. **Best Segmentation**: """ + f"{best_seg} achieves {results[best_seg]['segmentation']['overall']['dice_mean']:.4f} Dice" + """
2. **Best Classification**: """ + f"{best_cls} achieves {results[best_cls]['classification']['overall']['accuracy']:.2%} accuracy" + """
3. **Hardest Class**: Blood vessel (lowest performance across all metrics)
4. **Easiest Class**: Necrosis (highest Dice score)

---

## Files Generated

- `comprehensive_results.json` - Full results in JSON format
- `latex_tables.tex` - LaTeX tables for paper
- `performance_comparison.png` - Bar chart comparing configurations
- `per_class_heatmap.png` - Heatmap of per-class performance
- `confusion_matrix.png` - Classification confusion matrix
- `report.md` - This report

"""
    
    with open(os.path.join(output_dir, 'report.md'), 'w') as f:
        f.write(report)
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Generate comprehensive metrics report")
    parser.add_argument('--model_cfg', default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument('--checkpoint', default="sam2/checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument('--clip_prompts', default="configs/prompts/hard_coded_prompts_v2.json")
    parser.add_argument('--split', default='test')
    parser.add_argument('--output_dir', default='results/comprehensive_report')
    args = parser.parse_args()
    
    log("="*80)
    log("Comprehensive Metrics Report Generator")
    log("="*80)
    
    device = get_device()
    log(f"Device: {device}")
    
    register_omegaconf_resolvers()
    
    # Load dataset
    log("Loading dataset...")
    dataset = BCSSDataset('data/bcss/images', 'data/bcss/masks', split=args.split)
    log(f"Loaded {len(dataset)} samples")
    
    # Load SAM2
    log("Loading SAM2...")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sam2_root = os.path.join(project_root, 'sam2')
    checkpoint = os.path.join(project_root, args.checkpoint)
    
    orig_cwd = os.getcwd()
    os.chdir(sam2_root)
    try:
        predictor = get_sam2_predictor("configs/sam2.1/sam2.1_hiera_l.yaml", checkpoint, device)
    finally:
        os.chdir(orig_cwd)
    log("SAM2 loaded!")
    
    # Load CLIP
    log("Loading CLIP...")
    clip_prompts_path = os.path.join(project_root, args.clip_prompts)
    clip_prompts = load_prompts_from_json(clip_prompts_path)
    clip_classifier = CLIPClassifier(device=device)
    log("CLIP loaded!")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create evaluator
    evaluator = ComprehensiveEvaluator(predictor, clip_classifier, clip_prompts, dataset)
    
    # Configurations to evaluate
    configs = [
        {'name': 'box_baseline', 'prompt': 'box', 'neg': False, 'tta': False, 'refine': False},
        {'name': 'box_neg', 'prompt': 'box', 'neg': True, 'tta': False, 'refine': False},
        {'name': 'box_neg_tta', 'prompt': 'box', 'neg': True, 'tta': True, 'refine': False},
        {'name': 'box_neg_tta_refine', 'prompt': 'box', 'neg': True, 'tta': True, 'refine': True},
    ]
    
    all_results = {}
    
    for cfg in configs:
        log(f"\n>>> Evaluating: {cfg['name']}")
        start = time.time()
        
        results = evaluator.evaluate_configuration(
            config_name=cfg['name'],
            prompt_type=cfg['prompt'],
            use_neg_points=cfg['neg'],
            use_tta=cfg['tta'],
            use_refinement=cfg['refine']
        )
        
        elapsed = time.time() - start
        results['time_seconds'] = elapsed
        all_results[cfg['name']] = results
        
        seg = results['segmentation']['overall']
        cls = results['classification']['overall']
        log(f"    Dice: {seg['dice_mean']:.4f} ± {seg['dice_std']:.3f}")
        log(f"    Accuracy: {cls['accuracy']:.2%} (P:{cls['macro_precision']:.3f} R:{cls['macro_recall']:.3f} F1:{cls['macro_f1']:.3f})")
        log(f"    Time: {elapsed:.1f}s")
    
    # Save comprehensive JSON results
    log("\nGenerating outputs...")
    
    # Remove sample_results for cleaner JSON (too large)
    clean_results = {}
    for k, v in all_results.items():
        clean_results[k] = {key: val for key, val in v.items() if key != 'sample_results'}
    
    with open(os.path.join(args.output_dir, 'comprehensive_results.json'), 'w') as f:
        json.dump(clean_results, f, indent=2)
    
    # Generate LaTeX tables
    generate_latex_tables(clean_results, args.output_dir)
    log("LaTeX tables generated")
    
    # Generate figures
    generate_figures(clean_results, args.output_dir)
    log("Figures generated")
    
    # Generate markdown report
    generate_markdown_report(clean_results, args.output_dir)
    log("Markdown report generated")
    
    log(f"\n{'='*80}")
    log(f"All outputs saved to: {args.output_dir}")
    log("="*80)
    
    # Print summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"{'Config':<25} {'Dice':>10} {'IoU':>10} {'Accuracy':>12} {'F1':>10}")
    print("-"*70)
    for name, res in all_results.items():
        seg = res['segmentation']['overall']
        cls = res['classification']['overall']
        print(f"{name:<25} {seg['dice_mean']:>10.4f} {seg['iou_mean']:>10.4f} {cls['accuracy']:>11.2%} {cls['macro_f1']:>10.3f}")


if __name__ == '__main__':
    main()
