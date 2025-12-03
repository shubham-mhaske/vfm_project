#!/usr/bin/env python3
"""
Analyze all experiment results for presentation.
Generates summary tables and key findings.
"""

import os
import json
from datetime import datetime

def load_json(path):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except:
        return None

def main():
    results_dir = 'presentation_data/results'
    
    print("=" * 80)
    print("PROMPTABLE PATHOLOGY: EXPERIMENT RESULTS SUMMARY")
    print("=" * 80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print()
    
    # =========================================================================
    # 1. SAM2 SEGMENTATION RESULTS
    # =========================================================================
    print("\n" + "=" * 80)
    print("1. SAM2 SEGMENTATION RESULTS")
    print("=" * 80)
    
    # Full pipeline results
    pipeline_path = os.path.join(results_dir, 'full_pipeline_eval', 'full_pipeline_results.json')
    pipeline = load_json(pipeline_path)
    
    if pipeline:
        print("\n### SAM2 Configuration Ablation")
        print(f"{'Configuration':<30} {'Dice':>10} {'IoU':>10} {'vs Baseline':>12}")
        print("-" * 65)
        
        baseline_dice = None
        for config_name, config_data in pipeline.get('configurations', {}).items():
            dice = config_data.get('overall', {}).get('dice_mean', 0)
            iou = config_data.get('overall', {}).get('iou_mean', 0)
            
            if baseline_dice is None:
                baseline_dice = dice
                vs_baseline = "-"
            else:
                pct = (dice - baseline_dice) / baseline_dice * 100
                vs_baseline = f"+{pct:.1f}%" if pct > 0 else f"{pct:.1f}%"
            
            print(f"{config_name:<30} {dice:>10.4f} {iou:>10.4f} {vs_baseline:>12}")
    
    # Per-class breakdown
    print("\n### Best Config Per-Class (TTA + Refine)")
    print(f"{'Class':<20} {'Dice':>10} {'IoU':>10}")
    print("-" * 45)
    
    # Hardcoded from best results
    per_class = {
        'tumor': 0.5658,
        'stroma': 0.5096,
        'lymphocyte': 0.4814,
        'necrosis': 0.6076,
        'blood_vessel': 0.4823
    }
    for cls, dice in per_class.items():
        print(f"{cls:<20} {dice:>10.4f}")
    print(f"{'OVERALL':<20} {0.5256:>10.4f}")
    
    # =========================================================================
    # 2. CLIP CLASSIFICATION RESULTS
    # =========================================================================
    print("\n\n" + "=" * 80)
    print("2. CLIP CLASSIFICATION RESULTS")
    print("=" * 80)
    
    # Prompt comparison
    prompts_path = os.path.join(results_dir, 'clip_prompts_comparison', 'clip_prompts_comparison.json')
    prompts = load_json(prompts_path)
    
    if prompts:
        print("\n### CLIP Prompt Engineering")
        print(f"{'Prompt Config':<45} {'Accuracy':>12}")
        print("-" * 60)
        
        results_list = []
        for config_name, config_data in prompts.items():
            if isinstance(config_data, dict) and 'accuracy' in config_data:
                acc = config_data['accuracy']
                results_list.append((config_name, acc))
        
        # Sort by accuracy
        results_list.sort(key=lambda x: x[1], reverse=True)
        for name, acc in results_list:
            marker = " ✓ BEST" if acc == max(r[1] for r in results_list) else ""
            print(f"{name:<45} {acc*100:>11.1f}%{marker}")
    
    # Classifier results
    classifier_path = os.path.join(results_dir, 'clip_classifier', 'metrics.json')
    classifier = load_json(classifier_path)
    
    if classifier:
        print("\n### CLIP Feature Classifier")
        print(f"{'Method':<30} {'Accuracy':>12}")
        print("-" * 45)
        print(f"{'Zero-shot (fewshot prompts)':<30} {'38.9%':>12}")
        
        for method, data in classifier.get('results', {}).items():
            acc = data.get('accuracy', 0) * 100
            print(f"{method:<30} {acc:>11.1f}%")
    
    # PLIP comparison
    plip_path = os.path.join(results_dir, 'plip_test', 'metrics.json')
    plip = load_json(plip_path)
    
    if plip:
        print("\n### Medical CLIP (PLIP) Comparison")
        print(f"{'Model':<30} {'Accuracy':>12}")
        print("-" * 45)
        print(f"{'CLIP (general)':<30} {'38.9%':>12}")
        print(f"{'PLIP (pathology)':<30} {plip.get('overall_accuracy', 0)*100:>11.1f}%")
    
    # =========================================================================
    # 3. FAILED EXPERIMENTS (IMPORTANT FINDINGS)
    # =========================================================================
    print("\n\n" + "=" * 80)
    print("3. NEGATIVE RESULTS (Important Findings)")
    print("=" * 80)
    
    print("\n### SAM2 Fine-tuning (All Failed)")
    print(f"{'Approach':<30} {'Dice':>10} {'vs Zero-shot':>15}")
    print("-" * 58)
    
    finetune_results = [
        ("Zero-shot SAM2 (baseline)", 0.526, "-"),
        ("LoRA Light", 0.355, "-32.5%"),
        ("LoRA r=8", 0.266, "-49.4%"),
        ("Box + Focal Loss", 0.372, "-29.3%"),
    ]
    for name, dice, change in finetune_results:
        print(f"{name:<30} {dice:>10.3f} {change:>15}")
    
    print("\n→ Finding: Fine-tuning causes catastrophic forgetting with small dataset (85 images)")
    
    print("\n### Other Failed Approaches")
    print("- PLIP (medical CLIP): 26.9% vs CLIP 38.9%")
    print("- Ensemble CLIP+PLIP: 31.4% (worse than CLIP alone)")
    print("- Multi-scale crops: 25-28% (worse than single scale)")
    print("- Blur overlay + good prompts: Hurt performance")
    
    # =========================================================================
    # 4. KEY TAKEAWAYS
    # =========================================================================
    print("\n\n" + "=" * 80)
    print("4. KEY TAKEAWAYS FOR PRESENTATION")
    print("=" * 80)
    
    print("""
    1. ZERO-SHOT WINS: SAM2 zero-shot (0.526 Dice) beats all fine-tuning attempts
    
    2. PROMPT ENGINEERING MATTERS: 
       - SAM2: Box + Negative Points + TTA = +9.4% improvement
       - CLIP: Fewshot prompts = +3.3% over baseline
    
    3. CATASTROPHIC FORGETTING: Fine-tuning on small data (85 images) destroys 
       pretrained knowledge. Lightweight approaches (LoRA) also fail.
    
    4. DOMAIN-SPECIFIC ≠ BETTER: PLIP (pathology-trained) underperformed 
       general CLIP. May need prompt adaptation for new models.
    
    5. ENSEMBLE ≠ BETTER: Combining models didn't improve over best single model.
    
    BEST FINAL RESULTS:
    - Segmentation: 0.526 Dice (SAM2 + Box + Neg + TTA + Refine)
    - Classification: 40.4% (CLIP features + Logistic Regression)
    """)
    
    # =========================================================================
    # 5. PER-CLASS ANALYSIS
    # =========================================================================
    print("\n" + "=" * 80)
    print("5. PER-CLASS PERFORMANCE ANALYSIS")
    print("=" * 80)
    
    print(f"\n{'Class':<15} {'Seg Dice':>12} {'CLIP Acc':>12} {'Challenge':>30}")
    print("-" * 72)
    print(f"{'tumor':<15} {'0.566':>12} {'60.0%':>12} {'Good performance':>30}")
    print(f"{'stroma':<15} {'0.510':>12} {'64.4%':>12} {'Best classification':>30}")
    print(f"{'lymphocyte':<15} {'0.481':>12} {'2.7%':>12} {'Hard to classify (small)':>30}")
    print(f"{'necrosis':<15} {'0.608':>12} {'8.7%':>12} {'Best seg, poor classify':>30}")
    print(f"{'blood_vessel':<15} {'0.482':>12} {'23.3%':>12} {'Tiny structures (0.5%)':>30}")
    
    print("\n" + "=" * 80)
    print("END OF SUMMARY")
    print("=" * 80)


if __name__ == '__main__':
    main()
