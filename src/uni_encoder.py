"""
UNI Encoder Integration for Path-SAM2 Style Histopathology Segmentation

UNI (Universal) is a self-supervised vision transformer pretrained on 100M+ H&E 
histopathology patches from 100K+ whole-slide images covering 20 tissue types.

Reference: 
- Path-SAM2: https://arxiv.org/abs/2408.03651
- UNI: https://www.nature.com/articles/s41591-024-02857-3

To get UNI weights:
1. Request access at: https://huggingface.co/MahmoodLab/UNI
2. Download pytorch_model.bin
3. Place in: models/uni/pytorch_model.bin
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# Check if timm is available (required for UNI)
try:
    import timm
    from timm.models.vision_transformer import VisionTransformer
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm not installed. Install with: pip install timm")


class UNIEncoder(nn.Module):
    """
    UNI Encoder wrapper for histopathology feature extraction.
    
    UNI is a ViT-Large model pretrained on 100M+ H&E patches.
    Output: [B, 4096, 1024] for 1024x1024 input (64x64 patches, 1024 dim)
    """
    
    def __init__(
        self,
        checkpoint_path: str = "models/uni/pytorch_model.bin",
        freeze: bool = True,
        output_tokens: bool = True,  # Return patch tokens (not CLS)
    ):
        super().__init__()
        
        if not TIMM_AVAILABLE:
            raise ImportError("timm is required for UNI encoder. Install: pip install timm")
        
        self.output_tokens = output_tokens
        self.checkpoint_path = checkpoint_path
        
        # Create UNI model (ViT-Large with specific config)
        # UNI uses: vit_large_patch16_224 architecture scaled to larger images
        self.model = timm.create_model(
            "vit_large_patch16_224",
            img_size=1024,
            patch_size=16,
            init_values=1e-5,
            num_classes=0,  # No classification head
            dynamic_img_size=True,
        )
        
        # Load pretrained weights if available
        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_weights(checkpoint_path)
            print(f"✓ Loaded UNI weights from {checkpoint_path}")
        else:
            print(f"⚠ UNI weights not found at {checkpoint_path}")
            print("  Download from: https://huggingface.co/MahmoodLab/UNI")
        
        # Freeze encoder if specified
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
            print("✓ UNI encoder frozen")
    
    def _load_weights(self, checkpoint_path: str):
        """Load UNI pretrained weights."""
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        
        # Handle different checkpoint formats
        if "model" in state_dict:
            state_dict = state_dict["model"]
        
        # Load with strict=False to handle minor mismatches
        msg = self.model.load_state_dict(state_dict, strict=False)
        if msg.missing_keys:
            print(f"  Missing keys: {msg.missing_keys[:5]}...")
        if msg.unexpected_keys:
            print(f"  Unexpected keys: {msg.unexpected_keys[:5]}...")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through UNI encoder.
        
        Args:
            x: Input images [B, 3, H, W] (should be 1024x1024)
            
        Returns:
            Patch tokens [B, num_patches, embed_dim]
            For 1024x1024 input: [B, 4096, 1024] (64x64 patches, 1024 dim)
        """
        # Ensure input is 1024x1024
        if x.shape[-2] != 1024 or x.shape[-1] != 1024:
            x = F.interpolate(x, size=(1024, 1024), mode="bilinear", align_corners=False)
        
        # Get patch embeddings (excluding CLS token)
        # timm's forward_features returns [B, num_patches+1, embed_dim]
        features = self.model.forward_features(x)
        
        if self.output_tokens:
            # Remove CLS token, return patch tokens only
            # features shape: [B, 4097, 1024] -> [B, 4096, 1024]
            return features[:, 1:, :]
        else:
            # Return CLS token for classification
            return features[:, 0, :]
    
    @property
    def embed_dim(self) -> int:
        """Return embedding dimension (1024 for ViT-Large)."""
        return self.model.embed_dim
    
    @property
    def num_patches(self) -> int:
        """Return number of patches for 1024x1024 input."""
        return 64 * 64  # 4096


class DimensionAlignment(nn.Module):
    """
    Dimension alignment module to fuse SAM2 and UNI features.
    
    Following Path-SAM2: concatenate features then project to decoder dimension.
    SAM2 Hiera Large: 256 channels (after neck)
    UNI ViT-Large: 1024 channels
    Output: 256 channels (SAM2 decoder expects this)
    """
    
    def __init__(
        self,
        sam2_dim: int = 256,
        uni_dim: int = 1024,
        output_dim: int = 256,
    ):
        super().__init__()
        
        input_dim = sam2_dim + uni_dim  # 1280 for default
        
        # Two-layer projection with LayerNorm (following Path-SAM2)
        self.fusion = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False),
            nn.GroupNorm(32, output_dim),  # More stable than LayerNorm2d
            nn.GELU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, output_dim),
        )
    
    def forward(
        self, 
        sam2_features: torch.Tensor,
        uni_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse SAM2 and UNI features.
        
        Args:
            sam2_features: [B, C_sam, H, W] e.g., [B, 256, 64, 64]
            uni_features: [B, num_patches, C_uni] e.g., [B, 4096, 1024]
            
        Returns:
            Fused features [B, output_dim, H, W]
        """
        B, C_sam, H, W = sam2_features.shape
        
        # Reshape UNI features from [B, HW, C] to [B, C, H, W]
        # Assumes H*W = num_patches (64*64 = 4096)
        uni_features = uni_features.permute(0, 2, 1)  # [B, C, HW]
        uni_features = uni_features.view(B, -1, H, W)  # [B, C, H, W]
        
        # Concatenate along channel dimension
        fused = torch.cat([sam2_features, uni_features], dim=1)  # [B, C_sam+C_uni, H, W]
        
        # Project to output dimension
        return self.fusion(fused)


class PathSAM2Encoder(nn.Module):
    """
    Combined encoder following Path-SAM2 architecture.
    
    Combines:
    1. SAM2 Hiera backbone (general visual features)
    2. UNI encoder (histopathology-specific features)
    3. Dimension alignment (fusion module)
    
    This replaces SAM2's image encoder in the training pipeline.
    """
    
    def __init__(
        self,
        sam2_image_encoder: nn.Module,
        uni_checkpoint: str = "models/uni/pytorch_model.bin",
        freeze_uni: bool = True,
        freeze_sam2: bool = True,
        sam2_feature_dim: int = 256,
        uni_feature_dim: int = 1024,
        output_dim: int = 256,
    ):
        super().__init__()
        
        self.sam2_encoder = sam2_image_encoder
        self.uni_encoder = UNIEncoder(
            checkpoint_path=uni_checkpoint,
            freeze=freeze_uni,
        )
        self.dim_align = DimensionAlignment(
            sam2_dim=sam2_feature_dim,
            uni_dim=uni_feature_dim,
            output_dim=output_dim,
        )
        
        # Optionally freeze SAM2 encoder
        if freeze_sam2:
            for param in self.sam2_encoder.parameters():
                param.requires_grad = False
            print("✓ SAM2 encoder frozen")
    
    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass through combined encoder.
        
        Args:
            x: Input images [B, 3, H, W]
            
        Returns:
            Dictionary with fused vision features (compatible with SAM2)
        """
        # Get SAM2 features
        sam2_out = self.sam2_encoder(x)
        sam2_features = sam2_out["vision_features"]  # [B, 256, 64, 64]
        
        # Get UNI features
        uni_features = self.uni_encoder(x)  # [B, 4096, 1024]
        
        # Fuse features
        fused_features = self.dim_align(sam2_features, uni_features)
        
        # Return in SAM2-compatible format
        return {
            "vision_features": fused_features,
            "vision_pos_enc": sam2_out["vision_pos_enc"],
            "backbone_fpn": sam2_out["backbone_fpn"],
        }


def create_path_sam2_encoder(
    sam2_model,
    uni_checkpoint: str = "models/uni/pytorch_model.bin",
    freeze_uni: bool = True,
    freeze_sam2: bool = True,
) -> PathSAM2Encoder:
    """
    Factory function to create Path-SAM2 style encoder.
    
    Args:
        sam2_model: Loaded SAM2 model
        uni_checkpoint: Path to UNI weights
        freeze_uni: Whether to freeze UNI encoder
        freeze_sam2: Whether to freeze SAM2 encoder
        
    Returns:
        PathSAM2Encoder module
    """
    return PathSAM2Encoder(
        sam2_image_encoder=sam2_model.image_encoder,
        uni_checkpoint=uni_checkpoint,
        freeze_uni=freeze_uni,
        freeze_sam2=freeze_sam2,
    )


# ============================================================================
# Instructions for downloading UNI weights
# ============================================================================
"""
To download UNI weights:

1. Request access at Hugging Face:
   https://huggingface.co/MahmoodLab/UNI
   
2. Once approved, download:
   - pytorch_model.bin (~1.2GB)
   
3. Place in project:
   mkdir -p models/uni
   mv pytorch_model.bin models/uni/

4. Verify:
   python -c "from src.uni_encoder import UNIEncoder; e = UNIEncoder(); print('OK')"

Alternative: Use huggingface_hub
   pip install huggingface_hub
   from huggingface_hub import hf_hub_download
   hf_hub_download(repo_id="MahmoodLab/UNI", filename="pytorch_model.bin", local_dir="models/uni")
"""


if __name__ == "__main__":
    # Test the encoder
    print("Testing UNI Encoder...")
    
    # Check if weights exist
    uni_path = "models/uni/pytorch_model.bin"
    if not os.path.exists(uni_path):
        print(f"\n⚠ UNI weights not found at {uni_path}")
        print("Please download from: https://huggingface.co/MahmoodLab/UNI")
        print("\nTesting without pretrained weights...")
    
    # Create encoder
    try:
        encoder = UNIEncoder(checkpoint_path=uni_path, freeze=True)
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 1024, 1024)
        with torch.no_grad():
            output = encoder(dummy_input)
        
        print(f"\n✓ UNI Encoder test passed!")
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Expected: [1, 4096, 1024]")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
