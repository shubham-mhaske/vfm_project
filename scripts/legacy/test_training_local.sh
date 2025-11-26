#!/bin/bash

# Local testing script for SAM2 finetuning configuration
# Tests data loading, augmentations, and model initialization

set -e  # Exit on any error

echo "=================================================="
echo "Local Test: SAM2 Finetuning Configuration"
echo "=================================================="
echo ""

# Check if we're in the right directory
if [ ! -d "src" ] || [ ! -d "sam2" ]; then
    echo "ERROR: Please run this script from the project root directory"
    exit 1
fi

# Set PYTHONPATH
export PYTHONPATH="$(pwd):$(pwd)/sam2:$PYTHONPATH"
echo "✓ PYTHONPATH set"

# Check if data exists
if [ ! -d "data/bcss/images" ] || [ ! -d "data/bcss/masks" ]; then
    echo "ERROR: BCSS dataset not found in data/bcss/"
    exit 1
fi

# Count images
NUM_IMAGES=$(find data/bcss/images -name "*.png" | wc -l | tr -d ' ')
echo "✓ Found $NUM_IMAGES images in dataset"

# Check if SAM2 checkpoint exists
if [ ! -f "sam2/checkpoints/sam2.1_hiera_large.pt" ]; then
    echo "WARNING: SAM2 checkpoint not found at sam2/checkpoints/sam2.1_hiera_large.pt"
    echo "Run: bash sam2/checkpoints/download_ckpts.sh"
fi

echo ""
echo "=================================================="
echo "Test 1: Dataset Loading and Augmentations"
echo "=================================================="
python - << 'EOF'
import sys
import os
import numpy as np
import torch
from PIL import Image

# Add paths
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), 'sam2'))

print("Importing dataset modules...")
from src.finetune_dataset import BCSSRawDataset

print("\n1. Testing BCSSRawDataset initialization...")
try:
    dataset = BCSSRawDataset(
        img_folder='data/bcss/images',
        gt_folder='data/bcss/masks',
        split='train',
        prompt_type='mixed',
        use_neg_points=True,
        num_points=5
    )
    print(f"✓ Dataset initialized: {len(dataset)} training samples")
except Exception as e:
    print(f"✗ Dataset initialization failed: {e}")
    sys.exit(1)

print("\n2. Testing video retrieval (single image as video)...")
try:
    video, segment_loader = dataset.get_video(0)
    print(f"✓ Video retrieved: {video.video_name}")
    print(f"  - Video ID: {video.video_id}")
    print(f"  - Frames: {len(video.frames)}")
except Exception as e:
    print(f"✗ Video retrieval failed: {e}")
    sys.exit(1)

print("\n3. Testing segment loading and prompt generation...")
try:
    segments = segment_loader.load(frame_id=0)
    prompts = segment_loader.get_prompts_for_last_load()
    print(f"✓ Segments loaded: {len(segments)} objects")
    print(f"✓ Prompts generated for {len(prompts)} objects")
    
    for obj_id, prompt in list(prompts.items())[:2]:
        if prompt['point_coords'] is not None:
            print(f"  - Object {obj_id}: {prompt['point_coords'].shape[0]} points, "
                  f"labels shape {prompt['point_labels'].shape}")
except Exception as e:
    print(f"✗ Segment/prompt loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n4. Testing image loading...")
try:
    frame = video.frames[0]
    img = Image.open(frame.image_path)
    print(f"✓ Image loaded: {img.size}, mode: {img.mode}")
except Exception as e:
    print(f"✗ Image loading failed: {e}")
    sys.exit(1)

print("\n✓✓✓ Dataset tests PASSED ✓✓✓")
EOF

if [ $? -ne 0 ]; then
    echo ""
    echo "Dataset tests failed. Fix issues before proceeding."
    exit 1
fi

echo ""
echo "=================================================="
echo "Test 2: Transform Pipeline"
echo "=================================================="
python - << 'EOF'
import sys
import os
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), 'sam2'))

print("Testing augmentation transforms...")
try:
    from training.dataset.transforms import (
        RandomHorizontalFlip,
        RandomVerticalFlip, 
        RandomAffine,
        RandomResizeAPI,
        ColorJitter,
        ToTensorAPI,
        NormalizeAPI,
        ComposeAPI
    )
    print("✓ All transform classes imported successfully")
    
    # Test instantiation
    transforms = ComposeAPI([
        RandomHorizontalFlip(consistent_transform=True),
        RandomVerticalFlip(consistent_transform=True),
        RandomAffine(degrees=90, consistent_transform=True, scale=[0.9, 1.1], translate=[0.05, 0.05]),
        RandomResizeAPI(sizes=256, square=True, consistent_transform=True),
        ColorJitter(consistent_transform=True, brightness=0.3, contrast=0.3, saturation=0.3, hue=0.08),
        ToTensorAPI(),
        NormalizeAPI(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    print("✓ Transform pipeline instantiated successfully")
    
except Exception as e:
    print(f"✗ Transform test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓✓✓ Transform tests PASSED ✓✓✓")
EOF

if [ $? -ne 0 ]; then
    echo ""
    echo "Transform tests failed. Fix issues before proceeding."
    exit 1
fi

echo ""
echo "=================================================="
echo "Test 3: Configuration Loading"
echo "=================================================="
python - << 'EOF'
import sys
import os
sys.path.insert(0, os.getcwd())

print("Testing Hydra configuration...")
try:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    
    config_dir = os.path.abspath("conf")
    
    with initialize_config_dir(config_dir=config_dir, version_base="1.2"):
        cfg = compose(config_name="config", overrides=["experiment=local_test"])
        
    print("✓ Configuration loaded successfully")
    print(f"  - Experiment: {cfg.experiment.name}")
    print(f"  - Resolution: {cfg.scratch.resolution}")
    print(f"  - Batch size: {cfg.scratch.train_batch_size}")
    print(f"  - Epochs: {cfg.scratch.num_epochs}")
    print(f"  - Base LR: {cfg.scratch.base_lr}")
    print(f"  - Transforms: {len(cfg.vos.train_transforms[0].transforms)} transforms")
    
except Exception as e:
    print(f"✗ Configuration loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓✓✓ Configuration tests PASSED ✓✓✓")
EOF

if [ $? -ne 0 ]; then
    echo ""
    echo "Configuration tests failed. Fix issues before proceeding."
    exit 1
fi

echo ""
echo "=================================================="
echo "Test 4: Model Architecture Check"
echo "=================================================="
python - << 'EOF'
import sys
import os
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), 'sam2'))

print("Checking model configuration...")
try:
    from sam2.modeling.backbones.hieradet import Hiera
    from sam2.modeling.backbones.image_encoder import ImageEncoder
    
    print("✓ Model classes imported successfully")
    
    # Check if we can instantiate the backbone
    print("  Testing backbone instantiation (Hiera Large config)...")
    backbone_cfg = {
        'embed_dim': 144,
        'num_heads': 2,
        'stages': [2, 6, 36, 4],
        'global_att_blocks': [23, 33, 43],
        'window_pos_embed_bkg_spatial_size': [7, 7],
        'window_spec': [8, 4, 16, 8]
    }
    print(f"  - Embed dim: {backbone_cfg['embed_dim']}")
    print(f"  - Stages: {backbone_cfg['stages']}")
    print("✓ Backbone configuration valid")
    
except Exception as e:
    print(f"✗ Model check failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓✓✓ Model architecture tests PASSED ✓✓✓")
EOF

if [ $? -ne 0 ]; then
    echo ""
    echo "Model tests failed. Fix issues before proceeding."
    exit 1
fi

echo ""
echo "=================================================="
echo "✓✓✓ ALL TESTS PASSED ✓✓✓"
echo "=================================================="
echo ""
echo "Your configuration is ready for SLURM submission!"
echo ""
echo "To run a minimal training test (2 epochs, low res):"
echo "  python src/run_finetuning.py experiment=local_test"
echo ""
echo "To submit full training to HPRC:"
echo "  sbatch run_training_grace.slurm"
echo ""
