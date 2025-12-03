"""
Train SAM2 with LoRA adapters for histopathology segmentation.

This script implements parameter-efficient fine-tuning using LoRA (Low-Rank Adaptation).
Instead of modifying all 224M SAM2 parameters, we add small trainable LoRA matrices
(~0.2-1% of params) while keeping the original weights frozen.

Key benefits:
1. Prevents catastrophic forgetting of pretrained knowledge
2. Requires less GPU memory
3. Faster training
4. Can save/load just the LoRA weights (~5MB vs 1GB+)

Usage:
    python src/train_with_lora.py --lora_rank 8 --lr 1e-4 --epochs 20
"""

import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Add project paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sam2_root = os.path.join(project_root, 'sam2')
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if sam2_root not in sys.path:
    sys.path.insert(0, sam2_root)

from src.lora_adapter import apply_lora_to_sam2
from src.focal_loss import FocalDiceLoss


class BCSSLoRADataset(torch.utils.data.Dataset):
    """Simple BCSS dataset for LoRA training with proper resizing."""
    
    def __init__(self, image_dir, mask_dir, split='train', image_size=1024):
        from PIL import Image
        import numpy as np
        
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        
        all_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        
        # Define test set prefixes
        test_prefixes = ['TCGA-OL-', 'TCGA-LL-', 'TCGA-E2-', 'TCGA-EW-', 'TCGA-GM-', 'TCGA-S3-']
        test_files = [f for f in all_files if any(f.startswith(p) for p in test_prefixes)]
        train_val_files = [f for f in all_files if f not in test_files]
        
        np.random.seed(42)
        np.random.shuffle(train_val_files)
        val_split = int(0.2 * len(train_val_files))
        
        if split == 'train':
            self.image_files = train_val_files[val_split:]
        elif split == 'val':
            self.image_files = train_val_files[:val_split]
        else:
            raise ValueError(f"Invalid split: {split}")
        
        # Target classes
        self.target_class_ids = {1, 2, 3, 4, 18}
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        from PIL import Image
        import numpy as np
        import torchvision.transforms.functional as TF
        
        # Load and resize image
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        
        # Load and resize mask (using nearest neighbor to preserve class labels)
        mask_path = os.path.join(self.mask_dir, self.image_files[idx])
        mask = Image.open(mask_path)
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
        mask = np.array(mask)
        
        # Convert to tensors
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        mask = torch.from_numpy(mask).long()
        
        return {
            'image': image,
            'mask': mask,
            'filename': self.image_files[idx]
        }


def parse_args():
    parser = argparse.ArgumentParser(description='Train SAM2 with LoRA adapters')
    
    # Model configuration
    parser.add_argument('--sam_model_cfg', type=str, 
                        default='configs/sam2.1/sam2.1_hiera_l.yaml',
                        help='SAM2 config (relative to sam2 package)')
    parser.add_argument('--sam_checkpoint', type=str,
                        default='sam2/checkpoints/sam2.1_hiera_large.pt',
                        help='Path to SAM2 checkpoint')
    
    # LoRA configuration
    parser.add_argument('--lora_rank', type=int, default=8,
                        help='Rank of LoRA matrices (4, 8, or 16)')
    parser.add_argument('--lora_alpha', type=float, default=None,
                        help='LoRA scaling factor (default: same as rank)')
    parser.add_argument('--lora_dropout', type=float, default=0.1,
                        help='Dropout for LoRA layers')
    parser.add_argument('--target_modules', type=str, default='image_encoder',
                        choices=['image_encoder', 'mask_decoder', 'all'],
                        help='Which modules to apply LoRA to')
    parser.add_argument('--trainable_output_head', action='store_true', default=True,
                        help='Keep output MLP heads trainable')
    
    # Training configuration
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='Minimum learning rate for cosine annealing')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--warmup_epochs', type=int, default=2,
                        help='Number of warmup epochs')
    
    # Loss configuration
    parser.add_argument('--use_focal_loss', action='store_true', default=True,
                        help='Use FocalDiceLoss for class imbalance')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal loss gamma parameter')
    
    # Data configuration
    parser.add_argument('--data_dir', type=str, default='data/bcss',
                        help='Path to BCSS data directory')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Output configuration
    parser.add_argument('--output_dir', type=str, default='finetune_logs/lora',
                        help='Directory for outputs')
    parser.add_argument('--save_every', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--eval_every', type=int, default=2,
                        help='Evaluate every N epochs')
    
    # Hardware
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--amp', action='store_true', default=True,
                        help='Use automatic mixed precision')
    
    return parser.parse_args()


def setup_output_dir(args):
    """Create output directory with timestamp."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(project_root) / args.output_dir / f'lora_r{args.lora_rank}_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config = vars(args)
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    return output_dir


def load_sam2_model(args):
    """Load SAM2 model."""
    from sam2.build_sam import build_sam2
    
    # SAM2 uses Hydra which requires being in the sam2 directory for config resolution
    # Config path should be relative to sam2/sam2/configs/
    original_dir = os.getcwd()
    sam2_dir = os.path.join(project_root, 'sam2')
    
    try:
        os.chdir(sam2_dir)
        
        # Config path relative to sam2 package
        cfg_path = args.sam_model_cfg  # e.g., 'configs/sam2.1/sam2.1_hiera_l.yaml'
        ckpt_path = os.path.join(project_root, args.sam_checkpoint)
        
        print(f"Loading SAM2 from {ckpt_path}...")
        model = build_sam2(cfg_path, ckpt_path, device=args.device)
    finally:
        os.chdir(original_dir)
    
    return model


def create_dataloaders(args):
    """Create train and validation dataloaders."""
    data_dir = os.path.join(project_root, args.data_dir)
    image_dir = os.path.join(data_dir, 'images')
    mask_dir = os.path.join(data_dir, 'masks')
    
    train_dataset = BCSSLoRADataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        split='train',
        image_size=1024
    )
    
    val_dataset = BCSSLoRADataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        split='val',
        image_size=1024
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def compute_dice(pred, target, smooth=1e-6):
    """Compute Dice coefficient."""
    pred = pred.float()
    target = target.float()
    
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


class SimpleDiceBCELoss(nn.Module):
    """Simple Dice + BCE loss for mask prediction."""
    def __init__(self, dice_weight=1.0, bce_weight=1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
    
    def forward(self, pred, target):
        # BCE loss
        bce = F.binary_cross_entropy_with_logits(pred, target)
        
        # Dice loss
        pred_prob = torch.sigmoid(pred)
        intersection = (pred_prob * target).sum()
        union = pred_prob.sum() + target.sum()
        dice_loss = 1 - (2 * intersection + 1e-6) / (union + 1e-6)
        
        return self.bce_weight * bce + self.dice_weight * dice_loss


def train_epoch(model, train_loader, optimizer, criterion, device, scaler=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        # Create binary mask (any class > 0)
        binary_mask = (masks > 0).float().unsqueeze(1)  # (B, 1, H, W)
        
        optimizer.zero_grad()
        
        # Forward pass with AMP
        with torch.amp.autocast(device_type=device.type if hasattr(device, 'type') else 'cpu', enabled=scaler is not None):
            # Get image embeddings (LoRA is applied here)
            backbone_out = model.sam2_model.forward_image(images)
            
            # Get features from the backbone
            backbone_features = backbone_out['backbone_fpn'][-1]
            
            # Simple upsampling to get mask prediction
            # In a full implementation, you'd use the SAM decoder
            pred_masks = F.interpolate(
                backbone_features.mean(dim=1, keepdim=True),
                size=masks.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            
            loss = criterion(pred_masks, binary_mask)
        
        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        # Metrics
        with torch.no_grad():
            pred_binary = (pred_masks > 0).float()
            dice = compute_dice(pred_binary, binary_mask)
        
        total_loss += loss.item()
        total_dice += dice.item()
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice': f'{dice.item():.4f}'
        })
    
    return total_loss / num_batches, total_dice / num_batches


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    num_batches = 0
    
    for batch in tqdm(val_loader, desc='Validating'):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        # Create binary mask
        binary_mask = (masks > 0).float().unsqueeze(1)
        
        # Forward pass
        backbone_out = model.sam2_model.forward_image(images)
        backbone_features = backbone_out['backbone_fpn'][-1]
        
        pred_masks = F.interpolate(
            backbone_features.mean(dim=1, keepdim=True),
            size=masks.shape[-2:],
            mode='bilinear',
            align_corners=False
        )
        
        loss = criterion(pred_masks, binary_mask)
        
        pred_binary = (pred_masks > 0).float()
        dice = compute_dice(pred_binary, binary_mask)
        
        total_loss += loss.item()
        total_dice += dice.item()
        num_batches += 1
    
    return total_loss / num_batches, total_dice / num_batches


def main():
    args = parse_args()
    
    # Setup
    output_dir = setup_output_dir(args)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model and apply LoRA
    print("\n=== Loading SAM2 Model ===")
    sam2_model = load_sam2_model(args)
    
    print("\n=== Applying LoRA Adapters ===")
    lora_model = apply_lora_to_sam2(
        sam2_model,
        r=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_modules=args.target_modules,
        trainable_output_head=args.trainable_output_head
    )
    lora_model = lora_model.to(device)
    
    # Create dataloaders
    print("\n=== Creating Dataloaders ===")
    train_loader, val_loader = create_dataloaders(args)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Loss function - use simple Dice + BCE for this simplified training
    criterion = SimpleDiceBCELoss(dice_weight=1.0, bce_weight=1.0)
    
    # Optimizer (only LoRA parameters)
    trainable_params = lora_model.get_trainable_params()
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs - args.warmup_epochs,
        eta_min=args.min_lr
    )
    
    # AMP scaler
    scaler = torch.amp.GradScaler() if args.amp and device.type == 'cuda' else None
    
    # Training loop
    print("\n=== Starting Training ===")
    best_dice = 0.0
    history = {'train_loss': [], 'train_dice': [], 'val_loss': [], 'val_dice': []}
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Warmup learning rate
        if epoch < args.warmup_epochs:
            warmup_lr = args.lr * (epoch + 1) / args.warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
            print(f"Warmup LR: {warmup_lr:.2e}")
        
        # Train
        train_loss, train_dice = train_epoch(
            lora_model, train_loader, optimizer, criterion, device, scaler
        )
        history['train_loss'].append(train_loss)
        history['train_dice'].append(train_dice)
        
        # Update scheduler after warmup
        if epoch >= args.warmup_epochs:
            scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")
        
        # Validate
        if (epoch + 1) % args.eval_every == 0:
            val_loss, val_dice = validate(lora_model, val_loader, criterion, device)
            history['val_loss'].append(val_loss)
            history['val_dice'].append(val_dice)
            print(f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
            
            # Save best model
            if val_dice > best_dice:
                best_dice = val_dice
                lora_model.save_lora_weights(output_dir / 'best_lora_weights.pt')
                print(f"New best model saved! Dice: {best_dice:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            lora_model.save_lora_weights(output_dir / f'lora_weights_epoch{epoch+1}.pt')
    
    # Save final model and history
    lora_model.save_lora_weights(output_dir / 'final_lora_weights.pt')
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n=== Training Complete ===")
    print(f"Best validation Dice: {best_dice:.4f}")
    print(f"Outputs saved to: {output_dir}")


if __name__ == '__main__':
    main()
