#!/usr/bin/env python3
"""
Generate comprehensive training analysis, ablation studies, and publication figures.
Run from project root: python scripts/generate_training_analysis.py
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# Color palette
COLORS = {
    'focal': '#2E86AB',
    'lora': '#A23B72', 
    'base': '#F18F01',
    'zeroshot': '#3A7D44',
    'train': '#2E86AB',
    'val': '#C73E1D',
    'classes': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3A7D44']
}


def load_training_logs():
    """Load all training logs."""
    project_root = Path(__file__).parent.parent
    logs_dir = project_root / 'finetune_logs'
    
    training_data = {}
    
    # SAM2 Box Focal (50 epochs)
    focal_path = logs_dir / 'sam2_focal_50ep' / 'logs' / 'train_stats.json'
    if focal_path.exists():
        training_data['focal'] = parse_training_log(focal_path)
        print(f"  Loaded Focal: {len(training_data['focal']['epochs'])} epochs")
    
    # SAM2 LoRA Light (30 epochs)
    lora_path = logs_dir / 'sam2_lora_30ep' / 'logs' / 'train_stats.json'
    if lora_path.exists():
        training_data['lora'] = parse_training_log(lora_path)
        print(f"  Loaded LoRA: {len(training_data['lora']['epochs'])} epochs")
    
    # Base Finetune v2 (100 epochs)
    base_path = logs_dir / 'sam2_base_100ep' / 'logs' / 'train_stats.json'
    if base_path.exists():
        training_data['base'] = parse_training_log(base_path)
        print(f"  Loaded Base: {len(training_data['base']['epochs'])} epochs")
    
    return training_data


def parse_training_log(path):
    """Parse JSONL training log."""
    data = {
        'epochs': [],
        'total_loss': [],
        'dice_loss': [],
        'focal_loss': [],
        'mask_loss': [],
        'iou_loss': []
    }
    
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            data['epochs'].append(entry.get('Trainer/epoch', 0))
            data['total_loss'].append(entry.get('Losses/train_all_loss', 0))
            data['dice_loss'].append(entry.get('Losses/train_all_dice_loss', 
                                               entry.get('Losses/train_all_loss_dice', 0)))
            data['focal_loss'].append(entry.get('Losses/train_all_focal_loss', 0))
            data['mask_loss'].append(entry.get('Losses/train_all_loss_mask', 0))
            data['iou_loss'].append(entry.get('Losses/train_all_loss_iou', 0))
    
    return data


def load_evaluation_results():
    """Load evaluation results for finetuned models."""
    project_root = Path(__file__).parent.parent
    results_dir = project_root / 'results'
    
    eval_data = {}
    
    # SAM2 Box Focal Epoch 15
    focal_path = results_dir / 'sam2_box_focal_epoch15_eval' / 'metrics.json'
    if focal_path.exists():
        with open(focal_path) as f:
            eval_data['focal_e15'] = json.load(f)
    
    # SAM2 LoRA Light
    lora_path = results_dir / 'sam2_lora_light_eval' / 'metrics.json'
    if lora_path.exists():
        with open(lora_path) as f:
            eval_data['lora'] = json.load(f)
    
    # Load complete metrics for zero-shot comparison
    metrics_dir = results_dir / 'complete_metrics'
    sam2_files = sorted(list(metrics_dir.glob('sam2_segmentation_*.json')))
    if sam2_files:
        with open(sam2_files[-1]) as f:
            eval_data['zeroshot'] = json.load(f)
    
    return eval_data


def fig_training_curves(training_data, output_dir):
    """Plot training loss curves for all models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Total Loss
    ax1 = axes[0]
    if 'focal' in training_data:
        ax1.plot(training_data['focal']['epochs'], training_data['focal']['total_loss'], 
                 label='SAM2 Focal (50 ep)', color=COLORS['focal'], linewidth=2)
    if 'lora' in training_data:
        ax1.plot(training_data['lora']['epochs'], training_data['lora']['total_loss'],
                 label='SAM2 LoRA (30 ep)', color=COLORS['lora'], linewidth=2)
    if 'base' in training_data:
        ax1.plot(training_data['base']['epochs'], training_data['base']['total_loss'],
                 label='SAM2 Base (100 ep)', color=COLORS['base'], linewidth=2)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Total Loss')
    ax1.set_title('Training Loss Curves')
    ax1.legend()
    ax1.set_ylim(0, 60)
    
    # Dice Loss
    ax2 = axes[1]
    if 'focal' in training_data:
        ax2.plot(training_data['focal']['epochs'], training_data['focal']['dice_loss'],
                 label='SAM2 Focal', color=COLORS['focal'], linewidth=2)
    if 'lora' in training_data:
        ax2.plot(training_data['lora']['epochs'], training_data['lora']['dice_loss'],
                 label='SAM2 LoRA', color=COLORS['lora'], linewidth=2)
    if 'base' in training_data:
        ax2.plot(training_data['base']['epochs'], training_data['base']['dice_loss'],
                 label='SAM2 Base', color=COLORS['base'], linewidth=2)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Dice Loss')
    ax2.set_title('Dice Loss During Training')
    ax2.legend()
    ax2.set_ylim(0.6, 1.0)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig_training_curves.png')
    plt.savefig(output_dir / 'fig_training_curves.pdf')
    plt.close()
    print("✓ Training curves saved")


def fig_zeroshot_vs_finetuned(eval_data, output_dir):
    """Compare zero-shot vs finetuned performance."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = ['SAM2\nZero-Shot\n(Box+Neg)', 'SAM2\nFocal\n(Epoch 15)', 'SAM2\nLoRA\n(Epoch 30)']
    
    # Get overall dice scores
    scores = []
    if 'zeroshot' in eval_data:
        scores.append(eval_data['zeroshot']['box_neg_points']['overall']['dice_mean'])
    else:
        scores.append(0.555)  # Known value
    
    if 'focal_e15' in eval_data:
        scores.append(eval_data['focal_e15']['overall'])
    else:
        scores.append(0.372)  # Known value
    
    if 'lora' in eval_data:
        scores.append(eval_data['lora']['overall'])
    else:
        scores.append(0.355)  # Known value
    
    colors = [COLORS['zeroshot'], COLORS['focal'], COLORS['lora']]
    bars = ax.bar(models, scores, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Add horizontal line for zero-shot baseline
    ax.axhline(y=scores[0], color=COLORS['zeroshot'], linestyle='--', 
               linewidth=2, alpha=0.7, label='Zero-Shot Baseline')
    
    ax.set_ylabel('Dice Score')
    ax.set_title('Zero-Shot vs Finetuned Performance')
    ax.set_ylim(0, 0.7)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig_zeroshot_vs_finetuned.png')
    plt.savefig(output_dir / 'fig_zeroshot_vs_finetuned.pdf')
    plt.close()
    print("✓ Zero-shot vs Finetuned comparison saved")


def fig_perclass_finetuned(eval_data, output_dir):
    """Per-class comparison of finetuned models."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    classes = ['Tumor', 'Stroma', 'Lymphocyte', 'Necrosis', 'Blood Vessel']
    class_keys = ['tumor', 'stroma', 'lymphocyte', 'necrosis', 'blood_vessel']
    
    x = np.arange(len(classes))
    width = 0.25
    
    # Zero-shot
    if 'zeroshot' in eval_data:
        zs_scores = [eval_data['zeroshot']['box_neg_points']['per_class'][k]['dice_mean'] 
                     for k in class_keys]
    else:
        zs_scores = [0.560, 0.538, 0.532, 0.691, 0.497]
    
    # Focal
    if 'focal_e15' in eval_data:
        focal_scores = [eval_data['focal_e15'][k]['dice'] for k in class_keys]
    else:
        focal_scores = [0.386, 0.386, 0.301, 0.478, 0.335]
    
    # LoRA
    if 'lora' in eval_data:
        lora_scores = [eval_data['lora'][k]['dice'] for k in class_keys]
    else:
        lora_scores = [0.320, 0.394, 0.265, 0.498, 0.350]
    
    bars1 = ax.bar(x - width, zs_scores, width, label='Zero-Shot', color=COLORS['zeroshot'])
    bars2 = ax.bar(x, focal_scores, width, label='Focal Loss', color=COLORS['focal'])
    bars3 = ax.bar(x + width, lora_scores, width, label='LoRA', color=COLORS['lora'])
    
    ax.set_ylabel('Dice Score')
    ax.set_title('Per-Class Performance: Zero-Shot vs Finetuned')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()
    ax.set_ylim(0, 0.8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig_perclass_finetuned.png')
    plt.savefig(output_dir / 'fig_perclass_finetuned.pdf')
    plt.close()
    print("✓ Per-class finetuned comparison saved")


def fig_ablation_prompts(output_dir):
    """Ablation study on prompt types."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Data from complete metrics
    prompts = ['Centroid', 'Multi-Point', 'Box', 'Box+Neg']
    dice_scores = [0.338, 0.418, 0.553, 0.555]
    iou_scores = [0.236, 0.287, 0.407, 0.408]
    
    x = np.arange(len(prompts))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, dice_scores, width, label='Dice', color=COLORS['train'])
    bars2 = ax.bar(x + width/2, iou_scores, width, label='IoU', color=COLORS['val'])
    
    # Add value labels
    for bar, score in zip(bars1, dice_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontsize=10)
    for bar, score in zip(bars2, iou_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontsize=10)
    
    ax.set_ylabel('Score')
    ax.set_title('Ablation Study: Effect of Prompt Type on SAM2 Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(prompts)
    ax.legend()
    ax.set_ylim(0, 0.7)
    
    # Add improvement annotations
    ax.annotate('', xy=(3, 0.555), xytext=(0, 0.338),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.text(1.5, 0.58, '+64% improvement', ha='center', fontsize=11, color='gray')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig_ablation_prompts.png')
    plt.savefig(output_dir / 'fig_ablation_prompts.pdf')
    plt.close()
    print("✓ Prompt ablation study saved")


def fig_ablation_tta(output_dir):
    """Ablation study on Test-Time Augmentation."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Data
    configs = ['MedSAM\nBox', 'MedSAM\nBox + TTA', 'SAM2\nBox', 'SAM2\nBox + Neg']
    dice_scores = [0.522, 0.536, 0.553, 0.555]
    improvements = [0, 0.014, 0, 0.002]
    
    colors = [COLORS['lora'], COLORS['lora'], COLORS['train'], COLORS['train']]
    bars = ax.bar(configs, dice_scores, color=colors, edgecolor='black', linewidth=1)
    
    # Add improvement labels
    ax.annotate('+1.4%', xy=(1, 0.536), xytext=(1, 0.56),
                ha='center', fontsize=11, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='green'))
    ax.annotate('+0.2%', xy=(3, 0.555), xytext=(3, 0.58),
                ha='center', fontsize=11, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='green'))
    
    # Add value labels
    for bar, score in zip(bars, dice_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.03,
                f'{score:.3f}', ha='center', va='top', fontsize=11, color='white', fontweight='bold')
    
    ax.set_ylabel('Dice Score')
    ax.set_title('Ablation Study: Effect of TTA and Negative Points')
    ax.set_ylim(0, 0.65)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig_ablation_tta.png')
    plt.savefig(output_dir / 'fig_ablation_tta.pdf')
    plt.close()
    print("✓ TTA ablation study saved")


def fig_clip_prompts_ablation(output_dir):
    """Ablation study on CLIP prompt engineering."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Data sorted by accuracy
    prompts = [
        ('LLM Few-shot', 44.4, 'Text'),
        ('Hardcoded v2', 42.2, 'Manual'),
        ('LLM CLIP-opt', 35.6, 'Text'),
        ('LLM MM Few-shot', 29.4, 'Multimodal'),
        ('Hardcoded v1', 23.3, 'Manual'),
        ('LLM MM CLIP-opt', 15.0, 'Multimodal'),
        ('LLM Jargon', 12.2, 'Text'),
        ('LLM MM v1', 8.3, 'Multimodal')
    ]
    
    names = [p[0] for p in prompts]
    accs = [p[1] for p in prompts]
    types = [p[2] for p in prompts]
    
    color_map = {'Text': COLORS['train'], 'Manual': COLORS['zeroshot'], 'Multimodal': COLORS['lora']}
    colors = [color_map[t] for t in types]
    
    bars = ax.barh(names[::-1], accs[::-1], color=colors[::-1], edgecolor='black', linewidth=1)
    
    # Add value labels
    for bar, acc in zip(bars, accs[::-1]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{acc:.1f}%', ha='left', va='center', fontsize=10)
    
    ax.set_xlabel('Accuracy (%)')
    ax.set_title('Ablation: CLIP Prompt Engineering Strategies')
    ax.set_xlim(0, 55)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['train'], label='LLM Text'),
        Patch(facecolor=COLORS['zeroshot'], label='Manual'),
        Patch(facecolor=COLORS['lora'], label='LLM Multimodal')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig_ablation_clip_prompts.png')
    plt.savefig(output_dir / 'fig_ablation_clip_prompts.pdf')
    plt.close()
    print("✓ CLIP prompt ablation saved")


def fig_dataset_stats(output_dir):
    """Dataset statistics visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Class distribution
    ax1 = axes[0]
    classes = ['Tumor', 'Stroma', 'Lymphocyte', 'Necrosis', 'Blood Vessel']
    counts = [45, 45, 37, 23, 31]  # From test set per-class counts
    
    bars = ax1.bar(classes, counts, color=COLORS['classes'], edgecolor='black')
    ax1.set_ylabel('Sample Count (Test Set)')
    ax1.set_title('Class Distribution in Test Set')
    
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 str(count), ha='center', va='bottom', fontsize=11)
    
    # Data split
    ax2 = axes[1]
    splits = ['Train', 'Validation', 'Test']
    sizes = [85, 21, 45]
    colors_pie = [COLORS['train'], COLORS['val'], COLORS['zeroshot']]
    
    wedges, texts, autotexts = ax2.pie(sizes, labels=splits, autopct='%1.0f%%',
                                        colors=colors_pie, startangle=90)
    ax2.set_title('Dataset Split (151 images total)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig_dataset_stats.png')
    plt.savefig(output_dir / 'fig_dataset_stats.pdf')
    plt.close()
    print("✓ Dataset statistics saved")


def create_ablation_tables(output_dir):
    """Generate LaTeX tables for ablation studies."""
    
    # Table: Prompt Ablation
    table1 = r"""
\begin{table}[h]
\centering
\caption{Ablation Study: Effect of Prompt Type on SAM2 Segmentation}
\label{tab:prompt_ablation}
\begin{tabular}{lccccc}
\toprule
\textbf{Prompt Type} & \textbf{Dice} & \textbf{IoU} & \textbf{$\Delta$ Dice} & \textbf{Complexity} \\
\midrule
Centroid & 0.338 & 0.236 & - & Low \\
Multi-Point (5 pts) & 0.418 & 0.287 & +23.7\% & Medium \\
Bounding Box & 0.553 & 0.407 & +63.6\% & Medium \\
Box + Neg Points & \textbf{0.555} & \textbf{0.408} & \textbf{+64.2\%} & High \\
\bottomrule
\end{tabular}
\end{table}
"""

    # Table: TTA Ablation
    table2 = r"""
\begin{table}[h]
\centering
\caption{Ablation Study: Effect of Test-Time Augmentation}
\label{tab:tta_ablation}
\begin{tabular}{lcccc}
\toprule
\textbf{Model} & \textbf{Baseline} & \textbf{+ Enhancement} & \textbf{$\Delta$} \\
\midrule
MedSAM Box & 0.522 & 0.536 (TTA) & +1.4\% \\
SAM2 Box & 0.553 & 0.555 (Neg Pts) & +0.2\% \\
\bottomrule
\end{tabular}
\end{table}
"""

    # Table: Training Configuration Comparison
    table3 = r"""
\begin{table}[h]
\centering
\caption{Training Configuration Comparison}
\label{tab:training_config}
\begin{tabular}{lcccccc}
\toprule
\textbf{Config} & \textbf{Loss} & \textbf{Epochs} & \textbf{LR} & \textbf{Params} & \textbf{Val Dice} & \textbf{Test Dice} \\
\midrule
Zero-Shot & - & - & - & 0 & - & \textbf{0.555} \\
SAM2 Focal & Focal+Dice & 50 & 1e-4 & 224M & 0.381 & 0.372 \\
SAM2 LoRA & BCE+Dice & 30 & 1e-4 & 4.2M & - & 0.355 \\
SAM2 Base & BCE+Dice & 100 & 1e-4 & 224M & - & 0.371 \\
\bottomrule
\end{tabular}
\end{table}
"""

    # Table: CLIP Prompt Ablation
    table4 = r"""
\begin{table}[h]
\centering
\caption{Ablation Study: CLIP Prompt Engineering}
\label{tab:clip_ablation}
\begin{tabular}{llcc}
\toprule
\textbf{Prompt Source} & \textbf{Strategy} & \textbf{Accuracy} & \textbf{F1 Macro} \\
\midrule
LLM (GPT-4) & Text + Few-shot & \textbf{44.4\%} & \textbf{0.338} \\
Manual & Hardcoded v2 & 42.2\% & 0.312 \\
LLM (GPT-4) & Text + CLIP-opt & 35.6\% & 0.270 \\
LLM (Gemini) & Multimodal + Few-shot & 29.4\% & 0.220 \\
Manual & Hardcoded v1 & 23.3\% & 0.138 \\
LLM (Gemini) & Multimodal + CLIP-opt & 15.0\% & 0.100 \\
LLM (GPT-4) & Text + Jargon & 12.2\% & 0.097 \\
LLM (Gemini) & Multimodal v1 & 8.3\% & 0.091 \\
\bottomrule
\end{tabular}
\end{table}
"""

    # Table: Per-Class Detailed Results
    table5 = r"""
\begin{table*}[t]
\centering
\caption{Per-Class Segmentation Results Across All Methods}
\label{tab:perclass_all}
\begin{tabular}{l|ccccc|c}
\toprule
\textbf{Method} & \textbf{Tumor} & \textbf{Stroma} & \textbf{Lymph.} & \textbf{Necrosis} & \textbf{Blood V.} & \textbf{Mean} \\
\midrule
SAM2 Box+Neg (ZS) & 0.560 & 0.538 & 0.532 & \textbf{0.691} & \textbf{0.497} & \textbf{0.555} \\
SAM2 Box (ZS) & 0.548 & 0.509 & \textbf{0.551} & 0.704 & 0.512 & 0.553 \\
MedSAM TTA (ZS) & \textbf{0.575} & 0.505 & 0.549 & 0.647 & 0.427 & 0.536 \\
MedSAM Box (ZS) & 0.573 & 0.486 & 0.542 & 0.615 & 0.407 & 0.522 \\
SAM2 Focal (FT) & 0.386 & 0.386 & 0.301 & 0.478 & 0.335 & 0.372 \\
SAM2 LoRA (FT) & 0.320 & 0.394 & 0.265 & 0.498 & 0.350 & 0.355 \\
\bottomrule
\end{tabular}
\end{table*}
"""
    
    with open(output_dir / 'ablation_tables.tex', 'w') as f:
        f.write("% Ablation Study Tables for VFM Project\n")
        f.write("% Auto-generated - do not edit manually\n\n")
        f.write(table1)
        f.write('\n\n')
        f.write(table2)
        f.write('\n\n')
        f.write(table3)
        f.write('\n\n')
        f.write(table4)
        f.write('\n\n')
        f.write(table5)
    
    print("✓ LaTeX ablation tables saved")


def fig_model_size_comparison(output_dir):
    """Compare model sizes and inference times."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Model sizes
    ax1 = axes[0]
    models = ['SAM2\nHiera-L', 'MedSAM\nViT-B', 'CLIP\nViT-B/32', 'SAM2 LoRA\n(4.2M trainable)']
    params = [224, 86, 151, 4.2]
    
    bars = ax1.bar(models, params, color=[COLORS['train'], COLORS['lora'], 
                                          COLORS['zeroshot'], COLORS['focal']],
                   edgecolor='black')
    ax1.set_ylabel('Parameters (Millions)')
    ax1.set_title('Model Size Comparison')
    
    for bar, p in zip(bars, params):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 f'{p}M', ha='center', va='bottom', fontsize=11)
    
    # Performance vs Size scatter
    ax2 = axes[1]
    models_data = [
        ('SAM2 (224M)', 224, 0.555),
        ('MedSAM (86M)', 86, 0.536),
        ('SAM2 Focal (224M)', 224, 0.372),
        ('SAM2 LoRA (4.2M)', 4.2, 0.355)
    ]
    
    for name, size, perf in models_data:
        ax2.scatter(size, perf, s=200, label=name, edgecolors='black', linewidth=1.5)
    
    ax2.set_xlabel('Parameters (Millions)')
    ax2.set_ylabel('Dice Score')
    ax2.set_title('Performance vs Model Size')
    ax2.legend(loc='lower right')
    ax2.set_xlim(-10, 250)
    ax2.set_ylim(0.3, 0.6)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig_model_size.png')
    plt.savefig(output_dir / 'fig_model_size.pdf')
    plt.close()
    print("✓ Model size comparison saved")


def main():
    project_root = Path(__file__).parent.parent
    output_dir = project_root / 'results' / 'figures'
    output_dir.mkdir(exist_ok=True)
    
    print("Loading training logs...")
    training_data = load_training_logs()
    
    print("\nLoading evaluation results...")
    eval_data = load_evaluation_results()
    
    print(f"\nGenerating figures to {output_dir}/")
    print("-" * 50)
    
    # Training analysis
    if training_data:
        fig_training_curves(training_data, output_dir)
    
    # Comparison figures
    fig_zeroshot_vs_finetuned(eval_data, output_dir)
    fig_perclass_finetuned(eval_data, output_dir)
    
    # Ablation studies
    fig_ablation_prompts(output_dir)
    fig_ablation_tta(output_dir)
    fig_clip_prompts_ablation(output_dir)
    
    # Dataset and model analysis
    fig_dataset_stats(output_dir)
    fig_model_size_comparison(output_dir)
    
    # LaTeX tables
    create_ablation_tables(output_dir)
    
    print("-" * 50)
    print(f"\n✓ All figures saved to {output_dir}/")
    print("\nNew files generated:")
    for f in sorted(output_dir.glob('fig_*')):
        print(f"  - {f.name}")
    print("  - ablation_tables.tex")


if __name__ == '__main__':
    main()
