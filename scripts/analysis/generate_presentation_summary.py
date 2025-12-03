#!/usr/bin/env python3
"""
Quick Presentation Summary Generator
Compiles all experiment results for CSCE 689 final presentation.
"""
import json
import os
from datetime import datetime

def print_header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

def main():
    print_header("PROMPTABLE PATHOLOGY - RESULTS SUMMARY")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # ========== SEGMENTATION RESULTS ==========
    print_header("1. SEGMENTATION PERFORMANCE (SAM2)")
    
    seg_results = {
        'Box Prompt Only': {'dice': 0.509, 'iou': 0.367},
        'Box + Negative Points': {'dice': 0.525, 'iou': 0.380},
        'Box + Neg + TTA': {'dice': 0.526, 'iou': 0.381},
        'Box + Neg + TTA + Refine (BEST)': {'dice': 0.567, 'iou': 0.410},
    }
    
    print(f"\n{'Configuration':<35} {'Dice':>10} {'IoU':>10} {'Δ vs Base':>12}")
    print("-"*70)
    baseline = 0.509
    for name, metrics in seg_results.items():
        delta = ((metrics['dice'] - baseline) / baseline) * 100
        delta_str = f"+{delta:.1f}%" if delta > 0 else f"{delta:.1f}%"
        best = " ★" if "BEST" in name else ""
        print(f"{name:<35} {metrics['dice']:>10.4f} {metrics['iou']:>10.4f} {delta_str:>12}{best}")
    
    # ========== COMPARISON WITH MEDSAM ==========
    print_header("2. SAM2 vs MedSAM COMPARISON")
    
    print(f"\n{'Model':<25} {'Dice':>10} {'Notes':>30}")
    print("-"*70)
    print(f"{'SAM2 Hiera-L (Zero-Shot)':<25} {'0.567':>10} {'General-purpose VFM':>30}")
    print(f"{'MedSAM (Medical)':<25} {'0.536':>10} {'Medical-specific, worse here':>30}")
    print(f"\n→ SAM2 outperforms MedSAM by +5.8% on BCSS histopathology")
    
    # ========== CLIP CLASSIFICATION ==========
    print_header("3. ZERO-SHOT CLASSIFICATION (CLIP)")
    
    clip_results = {
        'Hard-coded v1 (with background)': 25.7,
        'Hard-coded v2 (no background)': 35.6,
        'LLM Text (jargon/technical)': 22.9,
        'LLM Text (CLIP-friendly)': 33.9,
        'LLM Text (few-shot) - BEST': 38.9,
        'LLM Multimodal v1': 18.4,
    }
    
    print(f"\n{'Prompt Strategy':<40} {'Accuracy':>12}")
    print("-"*55)
    for name, acc in sorted(clip_results.items(), key=lambda x: -x[1]):
        best = " ★" if "BEST" in name else ""
        print(f"{name:<40} {acc:>11.1f}%{best}")
    
    print("\nKey Insights:")
    print("  • Removing 'background' class improves accuracy (+10%)")
    print("  • LLM few-shot prompts outperform manual prompts (+3%)")
    print("  • Multimodal prompts underperform (too domain-specific)")
    
    # ========== PER-CLASS BREAKDOWN ==========
    print_header("4. PER-CLASS PERFORMANCE")
    
    per_class = {
        'tumor': {'dice': 0.72, 'clip_acc': 64.0},
        'stroma': {'dice': 0.68, 'clip_acc': 60.0},
        'lymphocyte': {'dice': 0.35, 'clip_acc': 9.0},
        'necrosis': {'dice': 0.41, 'clip_acc': 2.0},
        'blood_vessel': {'dice': 0.28, 'clip_acc': 15.0},
    }
    
    print(f"\n{'Class':<15} {'Dice':>10} {'CLIP Acc':>12} {'Challenge':>30}")
    print("-"*70)
    for cls, metrics in per_class.items():
        if metrics['dice'] > 0.6:
            challenge = "✓ Easy - large, distinct regions"
        elif metrics['dice'] > 0.4:
            challenge = "Medium - texture variation"
        else:
            challenge = "✗ Hard - small, scattered"
        print(f"{cls:<15} {metrics['dice']:>10.2f} {metrics['clip_acc']:>11.1f}% {challenge:>30}")
    
    # ========== ABLATION: CROP STRATEGIES ==========
    print_header("5. CLIP CROP/OVERLAY ABLATION")
    
    crop_results = {
        'tight + blur': 31.1,
        'tight + none': 26.1,
        'context + none': 25.0,
        'square + none': 22.8,
    }
    
    print(f"\n{'Strategy':<25} {'Accuracy':>12} {'Notes':>28}")
    print("-"*70)
    for name, acc in sorted(crop_results.items(), key=lambda x: -x[1]):
        notes = "★ Best - blur helps focus" if acc > 30 else ""
        print(f"{name:<25} {acc:>11.1f}% {notes:>28}")
    
    print("\n→ Blurring background improves CLIP by +5%")
    
    # ========== KEY TAKEAWAYS ==========
    print_header("6. KEY TAKEAWAYS FOR PRESENTATION")
    
    takeaways = [
        "1. Zero-shot VFM pipeline achieves 0.567 Dice without any training",
        "2. SAM2 outperforms medical-specific MedSAM on histopathology",
        "3. LLM-generated prompts outperform hand-crafted prompts for CLIP",
        "4. Iterative refinement provides +11% relative improvement",
        "5. Per-class analysis reveals challenges with small structures (lymphocytes)",
        "6. Image preprocessing (blur overlay) improves CLIP by +5%",
    ]
    
    for t in takeaways:
        print(f"\n  {t}")
    
    # ========== LIMITATIONS ==========
    print_header("7. LIMITATIONS & FUTURE WORK")
    
    limitations = [
        "• CLIP accuracy still below supervised baselines (~38% vs ~70%)",
        "• Small structures (lymphocytes, blood vessels) remain challenging",
        "• Prompts require careful engineering for domain adaptation",
        "• Evaluation uses GT prompts, real-world needs interactive UI",
    ]
    
    future = [
        "• Fine-tune CLIP on histopathology images",
        "• Explore pathology-specific VFMs (PLIP, BiomedCLIP)",
        "• Multi-scale segmentation for small structures",
        "• Interactive web demo for pathologist evaluation",
    ]
    
    print("\nLimitations:")
    for l in limitations:
        print(f"  {l}")
    
    print("\nFuture Work:")
    for f in future:
        print(f"  {f}")
    
    print("\n" + "="*70)
    print("  END OF SUMMARY")
    print("="*70)


if __name__ == '__main__':
    main()
