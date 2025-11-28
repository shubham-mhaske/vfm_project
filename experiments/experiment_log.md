# Experiment Log

Quick reference for experiment results. See `../EXPERIMENTS.md` for full details.

## 🎉 Key Finding

**Zeroshot SAM2 with optimized prompts achieves the best results!**

**Best Configuration: Box + Negative Points + TTA = 0.566 Dice**

## Summary Table

| Version | Date | Overall Dice | vs Baseline | Status |
|---------|------|--------------|-------------|--------|
| **Box+Neg+TTA** | 2025-11-27 | **0.566** | **+9.4%** | ✅ **BEST** |
| Box+TTA | 2025-11-27 | 0.563 | +8.9% | ✅ |
| Box+NegPts | 2025-11-27 | 0.560 | +8.4% | ✅ |
| Box baseline | 2025-11-27 | 0.553 | +7.0% | ✅ |
| Multi-point | 2025-11-27 | 0.418 | -19.2% | ⚠️ |
| Centroid | 2025-11-27 | 0.335 | -35.1% | ❌ |
| v2 Finetuned | 2025-11-22 | 0.42 | -18.8% | ❌ |
| Path-SAM2+CTP | 2025-11-27 | 0.366 | -29.3% | ❌ |
| LoRA-Light | 2025-11-27 | 0.355 | -31.4% | ❌ |
| Box+Focal | 2025-11-27 | 0.372 | -28.1% | ❌ |
| LoRA Adapter r=8 | 2025-11-27 | 0.266 | -48.6% | ❌ Worst |

## Best Config Per-Class Results

| Class | Dice | Count |
|-------|------|-------|
| Necrosis | 0.708 | 23 |
| Tumor | 0.565 | 45 |
| Lymphocyte | 0.562 | 37 |
| Stroma | 0.540 | 45 |
| Blood Vessel | 0.504 | 31 |
| **Overall** | **0.566** | 181 |

## Quick Commands

```bash
# Best configuration (0.566 Dice)
python src/evaluate_segmentation.py \
  --model_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --checkpoint sam2/checkpoints/sam2.1_hiera_large.pt \
  --prompt_type box --use_neg_points --use_tta \
  --split test --output_dir results/best_config

# Run all prompt experiments
bash scripts/run_prompt_experiments.sh
```

## Key Insights

1. **Box prompts >> Point prompts** (0.55 vs 0.33-0.42)
2. **Negative points add ~1%** (helps SAM know what NOT to segment)
3. **TTA adds ~1%** (free performance, no training)
4. **ALL finetuning hurts** - catastrophic forgetting with 85 images
5. **Blood vessel hardest** - tiny structures (0.5% of pixels)

---

**Last Updated**: November 27, 2025
