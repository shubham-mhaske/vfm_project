#!/usr/bin/env python3
"""
Generate publication-ready comparison figures for the VFM project.
Run from project root: python scripts/generate_comparison_figures.py
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

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

# Color palette (colorblind-friendly)
COLORS = {
    'sam2': '#2E86AB',      # Blue
    'medsam': '#A23B72',    # Magenta
    'clip': '#F18F01',      # Orange
    'finetuned': '#C73E1D', # Red
    'classes': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3A7D44']
}

def load_metrics():
    """Load all metrics from JSON files."""
    project_root = Path(__file__).parent.parent
    metrics_dir = project_root / 'results' / 'complete_metrics'
    
    metrics = {}
    
    # Load SAM2 metrics
    sam2_files = sorted(list(metrics_dir.glob('sam2_segmentation_*.json')))
    if sam2_files:
        with open(sam2_files[-1]) as f:  # Use latest file
            metrics['sam2'] = json.load(f)
            print(f"  Loaded SAM2: {sam2_files[-1].name}")
    
    # Load CLIP metrics
    clip_files = sorted(list(metrics_dir.glob('clip_classification_*.json')))
    if clip_files:
        with open(clip_files[-1]) as f:
            metrics['clip'] = json.load(f)
            print(f"  Loaded CLIP: {clip_files[-1].name}")
    
    # Load MedSAM metrics
    medsam_files = sorted(list(metrics_dir.glob('medsam_segmentation_*.json')))
    if medsam_files:
        with open(medsam_files[-1]) as f:
            metrics['medsam'] = json.load(f)
            print(f"  Loaded MedSAM: {medsam_files[-1].name}")
    
    # Debug: print structure
    if 'medsam' in metrics:
        print(f"  MedSAM keys: {list(metrics['medsam'].keys())}")
    
    return metrics


def fig1_segmentation_comparison(metrics, output_dir):
    """Bar chart comparing segmentation methods."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Data
    methods = ['SAM2\nBox+Neg', 'SAM2\nBox', 'MedSAM\nBox+TTA', 'MedSAM\nBox', 
               'SAM2\nMulti-Pt', 'SAM2\nCentroid']
    
    dice_scores = [
        metrics['sam2']['box_neg_points']['overall']['dice_mean'],
        metrics['sam2']['box_baseline']['overall']['dice_mean'],
        metrics['medsam']['medsam_box_tta']['overall']['dice_mean'],
        metrics['medsam']['medsam_box']['overall']['dice_mean'],
        metrics['sam2']['multi_point']['overall']['dice_mean'],
        metrics['sam2']['centroid']['overall']['dice_mean']
    ]
    
    dice_stds = [
        metrics['sam2']['box_neg_points']['overall']['dice_std'],
        metrics['sam2']['box_baseline']['overall']['dice_std'],
        metrics['medsam']['medsam_box_tta']['overall']['dice_std'],
        metrics['medsam']['medsam_box']['overall']['dice_std'],
        metrics['sam2']['multi_point']['overall']['dice_std'],
        metrics['sam2']['centroid']['overall']['dice_std']
    ]
    
    colors = [COLORS['sam2'], COLORS['sam2'], COLORS['medsam'], COLORS['medsam'],
              COLORS['sam2'], COLORS['sam2']]
    alphas = [1.0, 0.7, 1.0, 0.7, 0.5, 0.3]
    
    bars = ax.bar(methods, dice_scores, yerr=dice_stds, capsize=5,
                  color=colors, alpha=1.0, edgecolor='black', linewidth=1)
    
    for bar, alpha in zip(bars, alphas):
        bar.set_alpha(alpha)
    
    ax.set_ylabel('Dice Score')
    ax.set_title('Segmentation Performance Comparison')
    ax.set_ylim(0, 0.8)
    
    # Add value labels
    for bar, score in zip(bars, dice_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{score:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=COLORS['sam2'], label='SAM2'),
                       Patch(facecolor=COLORS['medsam'], label='MedSAM')]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_segmentation_comparison.png')
    plt.savefig(output_dir / 'fig1_segmentation_comparison.pdf')
    plt.close()
    print("✓ Figure 1: Segmentation comparison saved")


def fig2_perclass_heatmap(metrics, output_dir):
    """Heatmap of per-class performance."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    classes = ['Tumor', 'Stroma', 'Lymphocyte', 'Necrosis', 'Blood Vessel']
    class_keys = ['tumor', 'stroma', 'lymphocyte', 'necrosis', 'blood_vessel']
    
    methods = ['SAM2 Box+Neg', 'SAM2 Box', 'MedSAM TTA', 'MedSAM Box']
    
    data = np.zeros((len(methods), len(classes)))
    
    # SAM2 Box+Neg
    for j, key in enumerate(class_keys):
        data[0, j] = metrics['sam2']['box_neg_points']['per_class'][key]['dice_mean']
    
    # SAM2 Box
    for j, key in enumerate(class_keys):
        data[1, j] = metrics['sam2']['box_baseline']['per_class'][key]['dice_mean']
    
    # MedSAM TTA
    for j, key in enumerate(class_keys):
        data[2, j] = metrics['medsam']['medsam_box_tta']['per_class'][key]['dice_mean']
    
    # MedSAM Box
    for j, key in enumerate(class_keys):
        data[3, j] = metrics['medsam']['medsam_box']['per_class'][key]['dice_mean']
    
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0.3, vmax=0.75)
    
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(methods)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(methods)
    
    # Add text annotations
    for i in range(len(methods)):
        for j in range(len(classes)):
            text = ax.text(j, i, f'{data[i, j]:.3f}',
                          ha='center', va='center', color='black', fontsize=11)
    
    ax.set_title('Per-Class Dice Score Heatmap')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Dice Score')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig2_perclass_heatmap.png')
    plt.savefig(output_dir / 'fig2_perclass_heatmap.pdf')
    plt.close()
    print("✓ Figure 2: Per-class heatmap saved")


def fig3_clip_comparison(metrics, output_dir):
    """Bar chart comparing CLIP prompt strategies."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Sort by accuracy
    prompt_configs = [
        ('LLM Text\nFew-shot', 'llm_text_v3_fewshot'),
        ('Hardcoded\nv2', 'hardcoded_v2'),
        ('LLM Text\nCLIP-opt', 'llm_text_v2_clip_friendly'),
        ('LLM Multimodal\nFew-shot', 'llm_multimodal_v3_fewshot'),
        ('Hardcoded\nv1', 'hardcoded_v1'),
        ('LLM Multimodal\nCLIP-opt', 'llm_multimodal_v2_clip_friendly'),
        ('LLM Text\nJargon', 'llm_text_v1_jargon'),
        ('LLM Multimodal\nv1', 'llm_multimodal_v1')
    ]
    
    names = [p[0] for p in prompt_configs]
    accuracies = [metrics['clip'][p[1]]['overall']['accuracy'] * 100 for p in prompt_configs]
    
    colors = [COLORS['clip'] if 'LLM' in n else COLORS['sam2'] for n in names]
    
    bars = ax.bar(names, accuracies, color=colors, edgecolor='black', linewidth=1)
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('CLIP Classification by Prompt Strategy')
    ax.set_ylim(0, 60)
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # Rotate x labels
    plt.xticks(rotation=45, ha='right')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=COLORS['clip'], label='LLM-generated'),
                       Patch(facecolor=COLORS['sam2'], label='Hardcoded')]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_clip_comparison.png')
    plt.savefig(output_dir / 'fig3_clip_comparison.pdf')
    plt.close()
    print("✓ Figure 3: CLIP comparison saved")


def fig4_prompt_effect(metrics, output_dir):
    """Line plot showing effect of prompt type on segmentation."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    classes = ['Tumor', 'Stroma', 'Lymphocyte', 'Necrosis', 'Blood Vessel']
    class_keys = ['tumor', 'stroma', 'lymphocyte', 'necrosis', 'blood_vessel']
    
    prompt_types = ['Centroid', 'Multi-Point', 'Box', 'Box+Neg']
    prompt_keys = ['centroid', 'multi_point', 'box_baseline', 'box_neg_points']
    
    x = np.arange(len(prompt_types))
    width = 0.15
    
    for i, (cls, key) in enumerate(zip(classes, class_keys)):
        values = []
        for pk in prompt_keys:
            values.append(metrics['sam2'][pk]['per_class'][key]['dice_mean'])
        ax.bar(x + i*width, values, width, label=cls, color=COLORS['classes'][i])
    
    ax.set_ylabel('Dice Score')
    ax.set_title('Effect of Prompt Type on Per-Class Performance')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(prompt_types)
    ax.legend(loc='upper left', ncol=2)
    ax.set_ylim(0, 0.8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig4_prompt_effect.png')
    plt.savefig(output_dir / 'fig4_prompt_effect.pdf')
    plt.close()
    print("✓ Figure 4: Prompt effect saved")


def fig5_summary_radar(metrics, output_dir):
    """Radar chart comparing SAM2 vs MedSAM across classes."""
    from math import pi
    
    classes = ['Tumor', 'Stroma', 'Lymphocyte', 'Necrosis', 'Blood Vessel']
    class_keys = ['tumor', 'stroma', 'lymphocyte', 'necrosis', 'blood_vessel']
    
    sam2_scores = [metrics['sam2']['box_neg_points']['per_class'][k]['dice_mean'] for k in class_keys]
    medsam_scores = [metrics['medsam']['medsam_box_tta']['per_class'][k]['dice_mean'] for k in class_keys]
    
    # Close the radar chart
    sam2_scores += sam2_scores[:1]
    medsam_scores += medsam_scores[:1]
    
    angles = [n / float(len(classes)) * 2 * pi for n in range(len(classes))]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    ax.plot(angles, sam2_scores, 'o-', linewidth=2, label='SAM2 Box+Neg', color=COLORS['sam2'])
    ax.fill(angles, sam2_scores, alpha=0.25, color=COLORS['sam2'])
    
    ax.plot(angles, medsam_scores, 's-', linewidth=2, label='MedSAM TTA', color=COLORS['medsam'])
    ax.fill(angles, medsam_scores, alpha=0.25, color=COLORS['medsam'])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(classes)
    ax.set_ylim(0, 0.8)
    
    ax.set_title('SAM2 vs MedSAM: Per-Class Comparison', y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig5_radar_comparison.png')
    plt.savefig(output_dir / 'fig5_radar_comparison.pdf')
    plt.close()
    print("✓ Figure 5: Radar comparison saved")


def create_latex_tables(metrics, output_dir):
    """Generate LaTeX tables for the paper."""
    
    # Table 1: Main segmentation results
    table1 = r"""
\begin{table}[h]
\centering
\caption{Segmentation Performance on BCSS Test Set (n=45 images)}
\label{tab:segmentation}
\begin{tabular}{llccc}
\toprule
\textbf{Method} & \textbf{Prompt} & \textbf{Dice} & \textbf{IoU} & \textbf{Std} \\
\midrule
SAM2 & Box+Neg & \textbf{0.555} & \textbf{0.408} & 0.193 \\
SAM2 & Box & 0.553 & 0.407 & 0.195 \\
MedSAM & Box+TTA & 0.536 & 0.389 & 0.191 \\
MedSAM & Box & 0.522 & 0.375 & 0.189 \\
SAM2 & Multi-Point & 0.418 & 0.287 & 0.209 \\
SAM2 & Centroid & 0.338 & 0.236 & 0.263 \\
\bottomrule
\end{tabular}
\end{table}
"""
    
    # Table 2: Per-class results
    table2 = r"""
\begin{table}[h]
\centering
\caption{Per-Class Segmentation Performance (SAM2 Box+Neg)}
\label{tab:perclass}
\begin{tabular}{lcccc}
\toprule
\textbf{Class} & \textbf{N} & \textbf{Dice} & \textbf{IoU} & \textbf{Std} \\
\midrule
Necrosis & 23 & \textbf{0.691} & \textbf{0.559} & 0.194 \\
Tumor & 45 & 0.560 & 0.403 & 0.147 \\
Stroma & 45 & 0.538 & 0.385 & 0.166 \\
Lymphocyte & 37 & 0.532 & 0.391 & 0.218 \\
Blood Vessel & 31 & 0.497 & 0.357 & 0.208 \\
\bottomrule
\end{tabular}
\end{table}
"""
    
    with open(output_dir / 'latex_tables.tex', 'w') as f:
        f.write(table1)
        f.write('\n\n')
        f.write(table2)
    
    print("✓ LaTeX tables saved")


def main():
    project_root = Path(__file__).parent.parent
    output_dir = project_root / 'results' / 'figures'
    output_dir.mkdir(exist_ok=True)
    
    print("Loading metrics...")
    metrics = load_metrics()
    
    if not metrics:
        print("Error: No metrics found. Run evaluation first.")
        return
    
    print(f"\nGenerating figures to {output_dir}/")
    print("-" * 50)
    
    fig1_segmentation_comparison(metrics, output_dir)
    fig2_perclass_heatmap(metrics, output_dir)
    fig3_clip_comparison(metrics, output_dir)
    fig4_prompt_effect(metrics, output_dir)
    fig5_summary_radar(metrics, output_dir)
    create_latex_tables(metrics, output_dir)
    
    print("-" * 50)
    print(f"\n✓ All figures saved to {output_dir}/")
    print("\nFiles generated:")
    for f in output_dir.glob('*'):
        print(f"  - {f.name}")


if __name__ == '__main__':
    main()
