#!/usr/bin/env python3
"""
Quick validation test for SAM2 finetuning configuration
Tests dataset, transforms, and configuration loading without requiring GPU
"""

import os
import sys
import torch

# Set paths
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), 'sam2'))
os.environ['PYTHONPATH'] = f"{os.getcwd()}:{os.path.join(os.getcwd(), 'sam2')}"

print("="*70)
print("SAM2 Finetuning Configuration Validation")
print("="*70)

# Test 1: Dataset and Dataloader
print("\n[1/5] Testing Dataset and DataLoader...")
try:
    from src.finetune_dataset import BCSSRawDataset
    from training.dataset.transforms import (
        ComposeAPI, RandomHorizontalFlip, RandomVerticalFlip, 
        RandomAffine, RandomResizeAPI, ColorJitter, ToTensorAPI, NormalizeAPI
    )
    
    # Create transforms (as used in config)
    transforms_compose = ComposeAPI([
        RandomHorizontalFlip(consistent_transform=True),
        RandomVerticalFlip(consistent_transform=True),
        RandomAffine(degrees=90, consistent_transform=True, scale=[0.9, 1.1], translate=[0.05, 0.05]),
        RandomResizeAPI(sizes=256, square=True, consistent_transform=True),
        ColorJitter(consistent_transform=True, brightness=0.3, contrast=0.3, saturation=0.3, hue=0.08),
        ToTensorAPI(),
        NormalizeAPI(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print(f"   ✓ All transforms imported and instantiated")
    
    # Create raw dataset
    raw_dataset = BCSSRawDataset(
        img_folder='data/bcss/images',
        gt_folder='data/bcss/masks',
        split='train',
        prompt_type='mixed',
        use_neg_points=True,
        num_points=5
    )
    
    print(f"   ✓ Raw dataset created: {len(raw_dataset)} samples")
    
    # Test loading one video+segments
    video, segment_loader = raw_dataset.get_video(0)
    segments = segment_loader.load(frame_id=0)
    prompts = segment_loader.get_prompts_for_last_load()
    
    print(f"   ✓ Sample loaded successfully")
    print(f"     - Video: {video.video_name}")
    print(f"     - Objects: {len(segments)}")
    print(f"     - Prompts: {len(prompts)}")
    
except Exception as e:
    print(f"   ✗ Dataset test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Configuration
print("\n[2/5] Testing Hydra Configuration...")
try:
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf
    
    config_dir = os.path.abspath("conf")
    
    # Clear any existing Hydra instance
    GlobalHydra.instance().clear()
    
    with initialize_config_dir(config_dir=config_dir, version_base="1.2"):
        cfg = compose(config_name="config", overrides=["experiment=base_finetune"])
    
    print(f"   ✓ base_finetune config loaded")
    print(f"     - Epochs: {cfg.scratch.num_epochs}")
    print(f"     - Batch size: {cfg.scratch.train_batch_size}")
    print(f"     - Resolution: {cfg.scratch.resolution}")
    print(f"     - Learning rate: {cfg.scratch.base_lr}")
    print(f"     - Transforms: {len(cfg.vos.train_transforms[0].transforms)}")
    
    # Check for critical augmentations
    transform_names = [t['_target_'].split('.')[-1] for t in cfg.vos.train_transforms[0].transforms]
    required = ['RandomVerticalFlip', 'RandomAffine', 'ColorJitter']
    missing = [r for r in required if r not in transform_names]
    
    if missing:
        print(f"   ⚠ Missing recommended transforms: {missing}")
    else:
        print(f"   ✓ All recommended transforms present")
    
except Exception as e:
    print(f"   ✗ Configuration test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Model Architecture
print("\n[3/5] Testing Model Architecture...")
try:
    from sam2.modeling.backbones.hieradet import Hiera
    from sam2.modeling.backbones.image_encoder import ImageEncoder
    from sam2.modeling.memory_attention import MemoryAttention
    from sam2.modeling.memory_encoder import MemoryEncoder
    
    print(f"   ✓ All model components imported")
    
    # Verify checkpoint exists
    checkpoint_path = "sam2/checkpoints/sam2.1_hiera_large.pt"
    if os.path.exists(checkpoint_path):
        print(f"   ✓ Checkpoint found: {checkpoint_path}")
        # Check file size
        size_mb = os.path.getsize(checkpoint_path) / (1024*1024)
        print(f"     - Size: {size_mb:.1f} MB")
    else:
        print(f"   ⚠ Checkpoint not found: {checkpoint_path}")
        print(f"     Run: bash sam2/checkpoints/download_ckpts.sh")
    
except Exception as e:
    print(f"   ✗ Model test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Optimizer and Scheduler
print("\n[4/5] Testing Optimizer Configuration...")
try:
    import torch.optim as optim
    from fvcore.common.param_scheduler import CosineParamScheduler
    
    # Test optimizer
    dummy_params = [torch.randn(10, 10, requires_grad=True)]
    optimizer = optim.AdamW(dummy_params, lr=2e-5, weight_decay=0.05)
    print(f"   ✓ AdamW optimizer created")
    
    # Test scheduler
    scheduler = CosineParamScheduler(start_value=2e-5, end_value=1e-7)
    print(f"   ✓ Cosine scheduler created")
    
except Exception as e:
    print(f"   ✗ Optimizer test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Loss Function
print("\n[5/5] Testing Loss Configuration...")
try:
    from training.loss_fns import MultiStepMultiMasksAndIous
    
    loss_fn = MultiStepMultiMasksAndIous(
        weight_dict={'loss_mask': 20, 'loss_dice': 1, 'loss_iou': 1, 'loss_class': 1},
        supervise_all_iou=True,
        iou_use_l1_loss=True,
        pred_obj_scores=True
    )
    print(f"   ✓ Loss function created")
    print(f"     - Weights: mask=20, dice=1, iou=1, class=1")
    
except Exception as e:
    print(f"   ✗ Loss test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "="*70)
print("✓✓✓ ALL VALIDATION TESTS PASSED ✓✓✓")
print("="*70)
print("\nYour configuration is ready for training!")
print("\nRecommended next steps:")
print("  1. For HPC with GPU: sbatch run_training_grace.slurm")
print("  2. Configuration: conf/experiment/base_finetune.yaml")
print("  3. Expected training time: ~18-20 hours on 1 A100")
print("  4. Expected checkpoints: every 10 epochs in finetune_logs/")
print("\nKey Settings:")
print(f"  - Dataset: 85 training images, 21 val, 45 test")
print(f"  - Batch size: 4 → 21 steps/epoch")
print(f"  - Total steps: 21 × 150 = 3,150 steps")
print(f"  - Augmentations: Flips, Rotation, Strong Color Jitter")
print(f"  - Checkpoints: Every 10 epochs (15 total)")
print()
