"""
Path-SAM2 Training with CTransPath
===================================

This script implements Path-SAM2 style training using CTransPath encoder 
instead of UNI. CTransPath is fully open-source with no access restrictions.

Architecture:
    SAM2 Hiera-L (256-dim) + CTransPath Swin (768-dim) → Fusion → Decoder

CTransPath: Swin Transformer pretrained on 15M TCGA histopathology patches

Usage:
    python src/run_path_sam2_ctranspath.py experiment=path_sam2_ctranspath
    
Requirements:
    - CTransPath weights: models/ctranspath/ctranspath.pth
    - SAM2 checkpoint: sam2/checkpoints/sam2.1_hiera_large.pt
"""

import os
import sys
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import logging
import torch
import torch.nn as nn

# Add project root and sam2 to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sam2_root = os.path.join(project_root, 'sam2')
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if sam2_root not in sys.path:
    sys.path.insert(0, sam2_root)

from src.ctranspath_encoder import PathSAM2CTransPathEncoder, CTransPathEncoder, CTransPathDimensionAlignment

try:
    from training.utils.train_utils import register_omegaconf_resolvers
    from training.trainer import Trainer
except ImportError as e:
    print(f"Could not import from training utils: {e}")
    raise e


# Monkey-patch the trainer to skip DDP wrapping for single-GPU
_original_setup_ddp = None

def _skip_ddp_setup(self, distributed_conf, accelerator):
    """Skip DDP wrapping for single-GPU training with CTransPath."""
    print("  [CTransPath] Skipping DDP wrapper for single-GPU training")
    # Don't wrap in DDP - just keep the model as-is
    pass

def patch_trainer_skip_ddp():
    """Patch the Trainer class to skip DDP setup."""
    global _original_setup_ddp
    _original_setup_ddp = Trainer._setup_ddp_distributed_training
    Trainer._setup_ddp_distributed_training = _skip_ddp_setup
    print("  [CTransPath] Patched trainer to skip DDP")

def restore_trainer_ddp():
    """Restore original DDP setup."""
    global _original_setup_ddp
    if _original_setup_ddp is not None:
        Trainer._setup_ddp_distributed_training = _original_setup_ddp
        print("  [CTransPath] Restored original DDP setup")


def get_unwrapped_model(model):
    """Unwrap model from DDP/FSDP if needed."""
    if hasattr(model, 'module'):
        return model.module
    return model


def integrate_ctranspath_encoder(model, ctranspath_checkpoint: str, 
                                  sam_embed_dim: int = 256,
                                  ctranspath_embed_dim: int = 768,
                                  freeze_ctranspath: bool = True,
                                  fusion_type: str = 'concat'):
    """
    Replace SAM2's image encoder with PathSAM2CTransPathEncoder.
    
    Args:
        model: SAM2Train model (may be DDP-wrapped)
        ctranspath_checkpoint: Path to CTransPath weights
        sam_embed_dim: SAM2 output dimension (256)
        ctranspath_embed_dim: CTransPath output dimension (768)
        freeze_ctranspath: Freeze CTransPath weights
        fusion_type: Fusion strategy (concat, add, attention)
    
    Returns:
        Modified model (unwrapped)
    """
    print(f"\n{'='*60}")
    print("Integrating CTransPath Encoder (Path-SAM2 Style)")
    print(f"{'='*60}")
    
    # Unwrap DDP if needed
    unwrapped_model = get_unwrapped_model(model)
    
    original_encoder = unwrapped_model.image_encoder
    print(f"Original SAM2 encoder: {type(original_encoder).__name__}")
    
    # Create CTransPath encoder
    print(f"Loading CTransPath from: {ctranspath_checkpoint}")
    ctranspath_encoder = CTransPathEncoder(
        checkpoint_path=ctranspath_checkpoint,
        freeze=freeze_ctranspath
    )
    
    # Create fusion module
    fusion_module = CTransPathDimensionAlignment(
        sam_dim=sam_embed_dim,
        ctranspath_dim=ctranspath_embed_dim,
        output_dim=sam_embed_dim,
        fusion_type=fusion_type
    )
    
    # Create combined encoder
    path_sam2_encoder = PathSAM2CTransPathEncoder(
        sam_encoder=original_encoder,
        ctranspath_checkpoint=ctranspath_checkpoint,
        sam_embed_dim=sam_embed_dim,
        ctranspath_embed_dim=ctranspath_embed_dim,
        freeze_sam=True,
        freeze_ctranspath=freeze_ctranspath,
        fusion_type=fusion_type
    )
    
    # Replace encoder on the unwrapped model
    unwrapped_model.image_encoder = path_sam2_encoder
    
    # Parameter summary
    total_params = sum(p.numel() for p in unwrapped_model.parameters())
    trainable_params = sum(p.numel() for p in unwrapped_model.parameters() if p.requires_grad)
    
    print(f"\nParameter Summary:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Frozen: {total_params - trainable_params:,}")
    print(f"  Trainable %: {100 * trainable_params / total_params:.2f}%")
    print(f"{'='*60}\n")
    
    # Return the original model (DDP wrapper if it was wrapped)
    return model


def setup_selective_training(model, train_fusion: bool = True, train_decoder: bool = True):
    """
    Configure which parts of the model to train.
    
    Strategy:
    - SAM2 encoder: FROZEN
    - CTransPath encoder: FROZEN
    - Fusion module: TRAINABLE
    - Decoder: TRAINABLE
    """
    print("Configuring selective training...")
    
    # Unwrap DDP if needed
    unwrapped_model = get_unwrapped_model(model)
    
    # Freeze all first
    for param in unwrapped_model.parameters():
        param.requires_grad = False
    
    # Unfreeze fusion
    if train_fusion and hasattr(unwrapped_model, 'image_encoder'):
        if hasattr(unwrapped_model.image_encoder, 'dimension_alignment'):
            print("  ✓ Unfreezing fusion (dimension_alignment)")
            for param in unwrapped_model.image_encoder.dimension_alignment.parameters():
                param.requires_grad = True
    
    # Unfreeze decoder components
    if train_decoder:
        components = ['memory_attention', 'memory_encoder', 'sam_mask_decoder', 'sam_prompt_encoder']
        for comp_name in components:
            if hasattr(unwrapped_model, comp_name):
                comp = getattr(unwrapped_model, comp_name)
                print(f"  ✓ Unfreezing {comp_name}")
                for param in comp.parameters():
                    param.requires_grad = True
    
    trainable = sum(p.numel() for p in unwrapped_model.parameters() if p.requires_grad)
    print(f"  Total trainable: {trainable:,}")
    
    return model


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main training function."""
    
    try:
        register_omegaconf_resolvers()
    except Exception as e:
        logging.warning(f"Could not register resolvers: {e}")

    print("="*60)
    print("PATH-SAM2 TRAINING (CTransPath)")
    print("SAM2 + CTransPath Fusion for Histopathology")
    print("="*60)
    
    print("\n--- Configuration ---")
    print(OmegaConf.to_yaml(cfg))
    
    # Distributed setup - only set if not already set by torchrun
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = str(12355)
    if "RANK" not in os.environ:
        os.environ["RANK"] = "0"
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = "0"
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = "1"
    
    print(f"\nDistributed Setup:")
    print(f"  RANK: {os.environ.get('RANK', 'N/A')}")
    print(f"  WORLD_SIZE: {os.environ.get('WORLD_SIZE', 'N/A')}")
    print(f"  LOCAL_RANK: {os.environ.get('LOCAL_RANK', 'N/A')}")
    
    # Patch trainer to skip DDP for single-GPU CTransPath training
    print("\nPatching trainer for single-GPU training...")
    patch_trainer_skip_ddp()

    # Instantiate trainer (now without DDP wrapping)
    print("\nInstantiating trainer...")
    trainer = instantiate(cfg.trainer, _recursive_=False)
    
    # Get CTransPath config
    ctranspath_cfg = cfg.get('ctranspath', {})
    ctranspath_ckpt = ctranspath_cfg.get('checkpoint', 'models/ctranspath/ctranspath.pth')
    ctranspath_freeze = ctranspath_cfg.get('freeze', True)
    ctranspath_dim = ctranspath_cfg.get('embed_dim', 768)
    fusion_type = ctranspath_cfg.get('fusion_type', 'concat')
    
    # Resolve path
    if not os.path.isabs(ctranspath_ckpt):
        ctranspath_ckpt = os.path.join(project_root, ctranspath_ckpt)
    
    # Integrate CTransPath
    if os.path.exists(ctranspath_ckpt):
        print(f"\n✓ CTransPath checkpoint found: {ctranspath_ckpt}")
        trainer.model = integrate_ctranspath_encoder(
            model=trainer.model,
            ctranspath_checkpoint=ctranspath_ckpt,
            sam_embed_dim=256,
            ctranspath_embed_dim=ctranspath_dim,
            freeze_ctranspath=ctranspath_freeze,
            fusion_type=fusion_type
        )
        
        trainer.model = setup_selective_training(
            model=trainer.model,
            train_fusion=True,
            train_decoder=True
        )
    else:
        print(f"\n⚠️  WARNING: CTransPath checkpoint not found: {ctranspath_ckpt}")
        print("Download with: python src/download_ctranspath_weights.py")
        print("Running standard SAM2 training instead...")

    # Resume logic
    resume_path = cfg.get('resume_checkpoint', None)
    if resume_path and os.path.isfile(resume_path):
        print(f"[Resume] Loading: {resume_path}")
        try:
            ckpt = torch.load(resume_path, map_location='cpu')
            model_state = ckpt.get('model') or ckpt.get('model_state_dict')
            if model_state:
                missing, unexpected = trainer.model.load_state_dict(model_state, strict=False)
                print(f"[Resume] Loaded (missing={len(missing)}, unexpected={len(unexpected)})")
            
            if hasattr(trainer, 'optimizer') and 'optimizer' in ckpt:
                trainer.optimizer.load_state_dict(ckpt['optimizer'])
                
            start_epoch = ckpt.get('epoch')
            if start_epoch is not None and hasattr(trainer, 'start_epoch'):
                trainer.start_epoch = int(start_epoch) + 1
        except Exception as e:
            print(f"[Resume] Error: {e}")

    # Run training
    print("\n" + "="*60)
    print("Starting Path-SAM2 (CTransPath) Training...")
    print("="*60 + "\n")
    
    trainer.run()
    
    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Output: {os.getcwd()}")
    print("="*60)


if __name__ == "__main__":
    main()
