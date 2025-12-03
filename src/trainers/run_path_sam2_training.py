"""
Path-SAM2 Training Script
========================

This script implements Path-SAM2 style training by:
1. Loading standard SAM2 model
2. Loading pre-trained UNI encoder for histopathology
3. Replacing image encoder with fused PathSAM2Encoder
4. Training only the fusion layers + decoder (both encoders frozen)

Expected improvement: Dice 0.42 -> 0.65-0.80 on BCSS dataset

Usage:
    python src/run_path_sam2_training.py experiment=path_sam2_uni_fusion

Requirements:
    - UNI checkpoint: models/uni/pytorch_model.bin
    - SAM2 checkpoint: sam2/checkpoints/sam2.1_hiera_large.pt
    - timm library (pip install timm)
"""

import os
import sys
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import logging
import torch

# Add project root and sam2 to python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sam2_root = os.path.join(project_root, 'sam2')
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if sam2_root not in sys.path:
    sys.path.insert(0, sam2_root)

# Import UNI encoder module
from src.uni_encoder import PathSAM2Encoder, UNIEncoder, DimensionAlignment

try:
    from training.utils.train_utils import register_omegaconf_resolvers
except ImportError as e:
    print(f"Could not import 'register_omegaconf_resolvers'. Make sure 'sam2' is in the PYTHONPATH.")
    raise e


def integrate_uni_encoder(model, uni_checkpoint_path: str, sam_embed_dim: int = 256, 
                          uni_embed_dim: int = 1024, freeze_uni: bool = True):
    """
    Replace SAM2's image encoder with PathSAM2Encoder that fuses SAM2 + UNI features.
    
    Args:
        model: SAM2Train model instance
        uni_checkpoint_path: Path to UNI checkpoint
        sam_embed_dim: SAM2 output embedding dimension (256 for Hiera-L)
        uni_embed_dim: UNI output embedding dimension (1024)
        freeze_uni: Whether to freeze UNI encoder weights
    
    Returns:
        Modified model with PathSAM2Encoder
    """
    print(f"\n{'='*60}")
    print("Integrating UNI Encoder (Path-SAM2 Style)")
    print(f"{'='*60}")
    
    # Store original image encoder
    original_encoder = model.image_encoder
    print(f"Original SAM2 encoder type: {type(original_encoder).__name__}")
    
    # Create UNI encoder
    print(f"Loading UNI encoder from: {uni_checkpoint_path}")
    uni_encoder = UNIEncoder(
        checkpoint_path=uni_checkpoint_path,
        freeze=freeze_uni
    )
    
    # Create dimension alignment (fusion) module
    fusion_module = DimensionAlignment(
        sam_dim=sam_embed_dim,
        uni_dim=uni_embed_dim,
        output_dim=sam_embed_dim  # Output same dim as SAM2 expects
    )
    
    # Create PathSAM2Encoder that wraps both encoders
    path_sam2_encoder = PathSAM2Encoder(
        sam_encoder=original_encoder,
        uni_encoder=uni_encoder,
        dimension_alignment=fusion_module
    )
    
    # Replace model's image encoder
    model.image_encoder = path_sam2_encoder
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nParameter Summary:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters: {total_params - trainable_params:,}")
    print(f"  Trainable ratio: {100 * trainable_params / total_params:.2f}%")
    print(f"{'='*60}\n")
    
    return model


def setup_selective_training(model, train_fusion: bool = True, train_decoder: bool = True):
    """
    Freeze/unfreeze specific parts of the model for Path-SAM2 training.
    
    Training strategy:
    - SAM2 encoder: FROZEN (pretrained)
    - UNI encoder: FROZEN (pretrained on histopathology)
    - Fusion layers: TRAINABLE (learn to combine features)
    - Decoder + mask prediction: TRAINABLE (adapt to BCSS)
    """
    print("Setting up selective training...")
    
    # First, freeze everything
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze fusion module (dimension alignment)
    if train_fusion and hasattr(model, 'image_encoder') and hasattr(model.image_encoder, 'dimension_alignment'):
        print("  - Unfreezing fusion (dimension alignment) layers")
        for param in model.image_encoder.dimension_alignment.parameters():
            param.requires_grad = True
    
    # Unfreeze decoder-related components
    if train_decoder:
        # Memory attention
        if hasattr(model, 'memory_attention'):
            print("  - Unfreezing memory attention")
            for param in model.memory_attention.parameters():
                param.requires_grad = True
        
        # Memory encoder
        if hasattr(model, 'memory_encoder'):
            print("  - Unfreezing memory encoder")
            for param in model.memory_encoder.parameters():
                param.requires_grad = True
        
        # SAM mask decoder
        if hasattr(model, 'sam_mask_decoder'):
            print("  - Unfreezing SAM mask decoder")
            for param in model.sam_mask_decoder.parameters():
                param.requires_grad = True
        
        # SAM prompt encoder
        if hasattr(model, 'sam_prompt_encoder'):
            print("  - Unfreezing SAM prompt encoder")
            for param in model.sam_prompt_encoder.parameters():
                param.requires_grad = True
    
    # Final count
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total trainable parameters: {trainable_params:,}")
    
    return model


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main function for Path-SAM2 training with UNI encoder integration.
    """
    # --- Setup ---
    try:
        register_omegaconf_resolvers()
    except Exception as e:
        logging.warning(f"Could not register OmegaConf resolvers: {e}")

    print("="*60)
    print("PATH-SAM2 TRAINING")
    print("SAM2 + UNI Encoder Fusion for Histopathology Segmentation")
    print("="*60)
    
    print("\n--- Configuration ---")
    print(OmegaConf.to_yaml(cfg))
    print("-"*40)

    # Distributed setup (single GPU for now)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(12355)
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"

    # --- Instantiate Trainer ---
    print("\nInstantiating trainer...")
    trainer = instantiate(cfg.trainer, _recursive_=False)
    
    # --- Get UNI config ---
    uni_cfg = cfg.get('uni', {})
    uni_checkpoint = uni_cfg.get('checkpoint', 'models/uni/pytorch_model.bin')
    uni_freeze = uni_cfg.get('freeze', True)
    uni_embed_dim = uni_cfg.get('embed_dim', 1024)
    
    # Resolve UNI checkpoint path
    if not os.path.isabs(uni_checkpoint):
        uni_checkpoint = os.path.join(project_root, uni_checkpoint)
    
    # --- Integrate UNI Encoder ---
    if os.path.exists(uni_checkpoint):
        print(f"\nUNI checkpoint found: {uni_checkpoint}")
        trainer.model = integrate_uni_encoder(
            model=trainer.model,
            uni_checkpoint_path=uni_checkpoint,
            sam_embed_dim=256,  # Hiera-L output dim
            uni_embed_dim=uni_embed_dim,
            freeze_uni=uni_freeze
        )
        
        # Setup selective training (only train fusion + decoder)
        trainer.model = setup_selective_training(
            model=trainer.model,
            train_fusion=True,
            train_decoder=True
        )
    else:
        print(f"\n⚠️  WARNING: UNI checkpoint not found at {uni_checkpoint}")
        print("Running standard SAM2 training without UNI encoder.")
        print("To download UNI weights, run:")
        print("  python src/download_uni_weights.py")
        print("-"*40)

    # --- Optional Resume Logic ---
    resume_path = cfg.get('resume_checkpoint', None)
    if resume_path and os.path.isfile(resume_path):
        print(f"[Resume] Loading checkpoint: {resume_path}")
        try:
            ckpt = torch.load(resume_path, map_location='cpu')
            model_state = ckpt.get('model') or ckpt.get('model_state_dict')
            if model_state:
                missing, unexpected = trainer.model.load_state_dict(model_state, strict=False)
                print(f"[Resume] Model state loaded (missing={len(missing)}, unexpected={len(unexpected)})")
            
            if hasattr(trainer, 'optimizer') and 'optimizer' in ckpt:
                try:
                    trainer.optimizer.load_state_dict(ckpt['optimizer'])
                    print("[Resume] Optimizer state loaded.")
                except Exception as e:
                    print(f"[Resume] Failed to load optimizer state: {e}")
            
            trainer_start_epoch = ckpt.get('epoch')
            if trainer_start_epoch is not None and hasattr(trainer, 'start_epoch'):
                trainer.start_epoch = int(trainer_start_epoch) + 1
                print(f"[Resume] Resuming from epoch {trainer_start_epoch}")
        except Exception as e:
            print(f"[Resume] Error loading checkpoint: {e}")

    # --- Run Training ---
    print("\n" + "="*60)
    print("Starting Path-SAM2 Training...")
    print("="*60 + "\n")
    
    trainer.run()
    
    print("\n" + "="*60)
    print("Training finished successfully!")
    print(f"Output: {os.getcwd()}")
    print("="*60)


if __name__ == "__main__":
    main()
