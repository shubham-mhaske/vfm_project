# Experiment Tracking

Track all finetuning experiments with configs and results.

## Quick Reference

| Experiment | Date | Best Epoch | Val Dice | Val IoU | Key Changes | Status |
|------------|------|------------|----------|---------|-------------|--------|
| base_finetune_v1 | 2025-11-21 | 100 | 0.3354 | 0.2232 | Baseline | ⚠️ Unstable |
| base_finetune_v2_stable | TBD | TBD | TBD | TBD | Higher LR (5e-5), warmup (250 steps), balanced loss (5:2:1:1), lower weight decay (0.01), milder ColorJitter (0.2) | ✅ Ready |

## After Training

Validate checkpoints:
```bash
bash experiments/<exp_name>/validate_all_checkpoints.sh
```

Update results in `<exp_name>/results_summary.md` and `experiment_log.md`
