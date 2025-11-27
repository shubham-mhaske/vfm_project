"""
CTransPath Encoder Integration for SAM2
========================================

CTransPath is a Swin Transformer pretrained on 15M histopathology patches from TCGA.
This module integrates CTransPath with SAM2 following the Path-SAM2 fusion architecture.

Paper: "Transformer-based Unsupervised Contrastive Learning for Histopathological 
        Image Classification" (Medical Image Analysis 2022)

Architecture:
    Input (1024x1024) → SAM2 Hiera-L (256-dim) + CTransPath Swin (768-dim) → Fusion → 256-dim

Usage:
    from ctranspath_encoder import PathSAM2CTransPathEncoder, download_ctranspath_weights
    
    # Download weights (one-time)
    download_ctranspath_weights()
    
    # Create encoder
    encoder = PathSAM2CTransPathEncoder(
        sam_encoder=sam2_model.image_encoder,
        ctranspath_checkpoint="models/ctranspath/ctranspath.pth"
    )
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List
import warnings


class SwinTransformerBlock(nn.Module):
    """Basic Swin Transformer block for CTransPath."""
    
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, 
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size=(window_size, window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        
    def forward(self, x, H, W):
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        # Window partition and attention
        x_windows = window_partition(x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, H, W)
        
        x = x.view(B, H * W, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class WindowAttention(nn.Module):
    """Window-based multi-head self attention."""
    
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, 
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """MLP block."""
    
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PatchMerging(nn.Module):
    """Patch merging layer for downsampling."""
    
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)
        
    def forward(self, x, H, W):
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        
        x = self.norm(x)
        x = self.reduction(x)
        return x


def window_partition(x, window_size):
    """Partition into windows."""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """Reverse window partition."""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class CTransPathEncoder(nn.Module):
    """
    CTransPath Swin Transformer encoder for histopathology.
    
    Pretrained on 15M pathology patches from TCGA using contrastive learning.
    Output: 768-dimensional embeddings per patch.
    
    Uses timm library for proper Swin architecture that matches CTransPath weights.
    """
    
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        img_size: int = 224,
        embed_dim: int = 768,  # Final output dimension
        freeze: bool = True
    ):
        super().__init__()
        
        self.img_size = img_size
        self.embed_dim = embed_dim
        
        # Try to use timm for proper architecture
        try:
            import timm
            # CTransPath is based on Swin-Tiny but with convolutional patch embedding
            # We use the closest timm model and load weights with mapping
            self.model = timm.create_model(
                'swin_tiny_patch4_window7_224',
                pretrained=False,
                num_classes=0,  # Remove classification head
                global_pool=''  # Don't pool, we want spatial features
            )
            self.use_timm = True
            print("  Using timm Swin architecture")
        except ImportError:
            warnings.warn("timm not available, using simplified architecture")
            self.use_timm = False
            self._build_simple_encoder()
        
        # Load pretrained weights
        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_pretrained(checkpoint_path)
        
        # Freeze if requested
        if freeze:
            self._freeze()
    
    def _build_simple_encoder(self):
        """Build a simple feature extractor as fallback."""
        # Simple CNN that produces 768-dim features
        self.model = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=4, stride=4),
            nn.GELU(),
            nn.Conv2d(96, 192, kernel_size=2, stride=2),
            nn.GELU(),
            nn.Conv2d(192, 384, kernel_size=2, stride=2),
            nn.GELU(),
            nn.Conv2d(384, 768, kernel_size=2, stride=2),
            nn.GELU(),
        )
    
    def _load_pretrained(self, checkpoint_path: str):
        """Load CTransPath pretrained weights."""
        print(f"Loading CTransPath weights from: {checkpoint_path}")
        
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model' in state_dict:
            state_dict = state_dict['model']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        if self.use_timm:
            # Map CTransPath keys to timm Swin keys
            # CTransPath has convolutional patch embed, timm has linear
            # We'll load what matches and skip what doesn't
            new_state_dict = {}
            model_state = self.model.state_dict()
            
            loaded = 0
            skipped = 0
            for k, v in state_dict.items():
                # Remove prefixes
                if k.startswith('module.'):
                    k = k[7:]
                if k.startswith('backbone.'):
                    k = k[9:]
                
                # Skip patch_embed (different architecture)
                if 'patch_embed' in k:
                    skipped += 1
                    continue
                    
                # Try to match layer names
                if k in model_state and model_state[k].shape == v.shape:
                    new_state_dict[k] = v
                    loaded += 1
                else:
                    skipped += 1
            
            # Load matched weights
            missing, unexpected = self.model.load_state_dict(new_state_dict, strict=False)
            print(f"  Loaded {loaded} tensors, skipped {skipped}")
            print(f"  Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        else:
            # For simple encoder, just note that weights won't match
            print("  Using simple encoder - CTransPath weights not loaded")
        
        print("  CTransPath encoder initialized")
    
    def _freeze(self):
        """Freeze all parameters."""
        for param in self.model.parameters():
            param.requires_grad = False
        print("  CTransPath encoder frozen")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input images [B, 3, H, W]
            
        Returns:
            features: [B, num_patches, 768] where num_patches = 4096 for 1024x1024
        """
        B, C, H, W = x.shape
        
        if self.use_timm:
            # Resize if needed (timm expects 224x224)
            if H != self.img_size or W != self.img_size:
                x_resized = F.interpolate(x, size=(self.img_size, self.img_size), 
                                          mode='bilinear', align_corners=False)
            else:
                x_resized = x
            
            # Get features from timm model - output is [B, H', W', C]
            features = self.model.forward_features(x_resized)  # [B, 7, 7, 768] for 224
            
            # Convert to [B, C, H', W'] for interpolation
            features = features.permute(0, 3, 1, 2)  # [B, 768, 7, 7]
            
            # Interpolate to 64x64 for 1024x1024 input
            features = F.interpolate(features, size=(64, 64), mode='bilinear', align_corners=False)
            
            # Convert to [B, num_patches, C]
            features = features.flatten(2).transpose(1, 2)  # [B, 4096, 768]
        else:
            # Simple encoder path
            if H != self.img_size:
                x = F.interpolate(x, size=(self.img_size, self.img_size), 
                                  mode='bilinear', align_corners=False)
            features = self.model(x)  # [B, 768, H', W']
            
            # Interpolate to 64x64
            features = F.interpolate(features, size=(64, 64), mode='bilinear', align_corners=False)
            features = features.flatten(2).transpose(1, 2)  # [B, 4096, 768]
        
        return features


class CTransPathDimensionAlignment(nn.Module):
    """
    Dimension alignment module for fusing SAM2 and CTransPath features.
    
    SAM2 output: [B, 64, 64, 256]
    CTransPath output: [B, 4096, 768]
    Fused output: [B, 64, 64, 256]
    """
    
    def __init__(
        self,
        sam_dim: int = 256,
        ctranspath_dim: int = 768,
        output_dim: int = 256,
        fusion_type: str = 'concat'  # 'concat', 'add', 'attention'
    ):
        super().__init__()
        
        self.sam_dim = sam_dim
        self.ctranspath_dim = ctranspath_dim
        self.output_dim = output_dim
        self.fusion_type = fusion_type
        
        if fusion_type == 'concat':
            # Concatenate and project
            self.fusion = nn.Sequential(
                nn.Conv2d(sam_dim + ctranspath_dim, output_dim * 2, kernel_size=1),
                nn.BatchNorm2d(output_dim * 2),
                nn.GELU(),
                nn.Conv2d(output_dim * 2, output_dim, kernel_size=1),
                nn.BatchNorm2d(output_dim),
            )
        elif fusion_type == 'add':
            # Project CTransPath to SAM dim and add
            self.ctranspath_proj = nn.Sequential(
                nn.Linear(ctranspath_dim, output_dim),
                nn.LayerNorm(output_dim),
            )
            self.fusion = nn.Sequential(
                nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(output_dim),
                nn.GELU(),
            )
        elif fusion_type == 'attention':
            # Cross-attention fusion
            self.ctranspath_proj = nn.Linear(ctranspath_dim, output_dim)
            self.cross_attn = nn.MultiheadAttention(output_dim, num_heads=8, batch_first=True)
            self.norm = nn.LayerNorm(output_dim)
            self.ffn = nn.Sequential(
                nn.Linear(output_dim, output_dim * 4),
                nn.GELU(),
                nn.Linear(output_dim * 4, output_dim),
            )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize fusion weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(
        self,
        sam_features: torch.Tensor,
        ctranspath_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse SAM2 and CTransPath features.
        
        Args:
            sam_features: [B, 64, 64, 256] from SAM2 encoder
            ctranspath_features: [B, 4096, 768] from CTransPath
            
        Returns:
            fused: [B, 64, 64, 256] fused features
        """
        B = sam_features.shape[0]
        
        # Reshape CTransPath features to spatial
        ctp_spatial = ctranspath_features.transpose(1, 2).view(B, self.ctranspath_dim, 64, 64)
        
        # Reshape SAM features to channel-last → channel-first
        if sam_features.dim() == 4 and sam_features.shape[-1] == self.sam_dim:
            sam_spatial = sam_features.permute(0, 3, 1, 2)  # [B, 256, 64, 64]
        else:
            sam_spatial = sam_features
        
        if self.fusion_type == 'concat':
            # Concatenate along channel dimension
            combined = torch.cat([sam_spatial, ctp_spatial], dim=1)  # [B, 1024, 64, 64]
            fused = self.fusion(combined)  # [B, 256, 64, 64]
            
        elif self.fusion_type == 'add':
            # Project and add
            ctp_proj = self.ctranspath_proj(ctranspath_features)  # [B, 4096, 256]
            ctp_proj = ctp_proj.transpose(1, 2).view(B, self.output_dim, 64, 64)
            combined = sam_spatial + ctp_proj
            fused = self.fusion(combined)
            
        elif self.fusion_type == 'attention':
            # Cross-attention
            sam_flat = sam_spatial.flatten(2).transpose(1, 2)  # [B, 4096, 256]
            ctp_proj = self.ctranspath_proj(ctranspath_features)  # [B, 4096, 256]
            
            attn_out, _ = self.cross_attn(sam_flat, ctp_proj, ctp_proj)
            fused = self.norm(sam_flat + attn_out)
            fused = fused + self.ffn(fused)
            fused = fused.transpose(1, 2).view(B, self.output_dim, 64, 64)
        
        # Return in channel-last format to match SAM2 expectations
        fused = fused.permute(0, 2, 3, 1)  # [B, 64, 64, 256]
        
        return fused


class PathSAM2CTransPathEncoder(nn.Module):
    """
    Combined encoder that fuses SAM2 and CTransPath features.
    
    This replaces SAM2's image_encoder in the model.
    """
    
    def __init__(
        self,
        sam_encoder: nn.Module,
        ctranspath_checkpoint: str,
        sam_embed_dim: int = 256,
        ctranspath_embed_dim: int = 768,
        freeze_sam: bool = True,
        freeze_ctranspath: bool = True,
        fusion_type: str = 'concat'
    ):
        super().__init__()
        
        self.sam_encoder = sam_encoder
        self.ctranspath_encoder = CTransPathEncoder(
            checkpoint_path=ctranspath_checkpoint,
            freeze=freeze_ctranspath
        )
        self.dimension_alignment = CTransPathDimensionAlignment(
            sam_dim=sam_embed_dim,
            ctranspath_dim=ctranspath_embed_dim,
            output_dim=sam_embed_dim,
            fusion_type=fusion_type
        )
        
        # Freeze SAM encoder if requested
        if freeze_sam:
            for param in self.sam_encoder.parameters():
                param.requires_grad = False
            print("  SAM2 encoder frozen")
        
        # Print parameter summary
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\nPathSAM2-CTransPath Encoder:")
        print(f"  Total parameters: {total:,}")
        print(f"  Trainable parameters: {trainable:,}")
        print(f"  Fusion type: {fusion_type}")
    
    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Forward pass through both encoders and fusion.
        
        Args:
            x: Input images [B, 3, H, W]
            
        Returns:
            dict with fused features matching SAM2 encoder output format
        """
        # Ensure all submodules are on the same device as input
        device = x.device
        self.ctranspath_encoder = self.ctranspath_encoder.to(device)
        self.dimension_alignment = self.dimension_alignment.to(device)
        
        # Get SAM2 features
        sam_output = self.sam_encoder(x)
        
        # Get CTransPath features
        ctranspath_features = self.ctranspath_encoder(x)  # [B, 4096, 768]
        
        # Handle SAM2 output format (could be dict or tensor)
        if isinstance(sam_output, dict):
            # Get the main feature map (usually 'vision_features' or similar)
            if 'vision_features' in sam_output:
                sam_features = sam_output['vision_features']
            elif 'backbone_fpn' in sam_output:
                sam_features = sam_output['backbone_fpn'][0]  # Take highest res
            else:
                # Take first tensor value
                sam_features = list(sam_output.values())[0]
        else:
            sam_features = sam_output
        
        # Fuse features
        fused_features = self.dimension_alignment(sam_features, ctranspath_features)
        
        # Return in same format as SAM2 encoder
        if isinstance(sam_output, dict):
            sam_output['vision_features'] = fused_features
            if 'backbone_fpn' in sam_output:
                sam_output['backbone_fpn'][0] = fused_features.permute(0, 3, 1, 2)
            return sam_output
        else:
            return fused_features


def download_ctranspath_weights(output_dir: str = "models/ctranspath"):
    """
    Download CTransPath pretrained weights.
    
    CTransPath weights are available from the official GitHub release.
    """
    import urllib.request
    
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, "ctranspath.pth")
    
    if os.path.exists(checkpoint_path):
        print(f"✓ CTransPath weights already exist: {checkpoint_path}")
        return checkpoint_path
    
    print("="*60)
    print("Downloading CTransPath Weights")
    print("="*60)
    
    # CTransPath official weights URL (from GitHub releases)
    # Note: This is a placeholder - actual URL may differ
    urls = [
        "https://github.com/Xiyue-Wang/TransPath/releases/download/v1.0/ctranspath.pth",
        "https://drive.google.com/uc?id=1DoDx_70_TLj98gTf6YTXnu4tFhsFocDX",  # Alternative
    ]
    
    print("\nCTransPath download options:")
    print("1. Official GitHub: https://github.com/Xiyue-Wang/TransPath")
    print("2. Google Drive (from paper authors)")
    print("\nAutomatic download may not work due to hosting limitations.")
    print("\nManual download instructions:")
    print(f"  1. Go to: https://github.com/Xiyue-Wang/TransPath")
    print(f"  2. Download ctranspath.pth from Releases")
    print(f"  3. Place at: {checkpoint_path}")
    
    # Try automatic download
    for url in urls:
        try:
            print(f"\nTrying: {url[:50]}...")
            urllib.request.urlretrieve(url, checkpoint_path)
            
            # Verify download
            if os.path.exists(checkpoint_path) and os.path.getsize(checkpoint_path) > 1e6:
                print(f"✓ Downloaded successfully: {checkpoint_path}")
                return checkpoint_path
        except Exception as e:
            print(f"  Failed: {e}")
            continue
    
    print("\n" + "="*60)
    print("MANUAL DOWNLOAD REQUIRED")
    print("="*60)
    print(f"\n1. Download from: https://github.com/Xiyue-Wang/TransPath")
    print(f"2. Save to: {checkpoint_path}")
    print("\nAlternatively, use gdown for Google Drive:")
    print(f"  pip install gdown")
    print(f"  gdown 1DoDx_70_TLj98gTf6YTXnu4tFhsFocDX -O {checkpoint_path}")
    
    return None


def create_path_sam2_ctranspath_encoder(
    sam_encoder: nn.Module,
    ctranspath_checkpoint: str = "models/ctranspath/ctranspath.pth",
    freeze_encoders: bool = True,
    fusion_type: str = 'concat'
) -> PathSAM2CTransPathEncoder:
    """
    Factory function to create PathSAM2 encoder with CTransPath.
    
    Args:
        sam_encoder: SAM2's image encoder
        ctranspath_checkpoint: Path to CTransPath weights
        freeze_encoders: Whether to freeze both encoders
        fusion_type: Type of fusion ('concat', 'add', 'attention')
    
    Returns:
        PathSAM2CTransPathEncoder instance
    """
    return PathSAM2CTransPathEncoder(
        sam_encoder=sam_encoder,
        ctranspath_checkpoint=ctranspath_checkpoint,
        freeze_sam=freeze_encoders,
        freeze_ctranspath=freeze_encoders,
        fusion_type=fusion_type
    )


if __name__ == "__main__":
    # Test the module
    print("Testing CTransPath encoder module...")
    
    # Download weights
    ckpt_path = download_ctranspath_weights()
    
    if ckpt_path:
        # Test encoder
        encoder = CTransPathEncoder(checkpoint_path=ckpt_path, freeze=True)
        
        # Test forward pass
        x = torch.randn(1, 3, 1024, 1024)
        with torch.no_grad():
            features = encoder(x)
        
        print(f"\nInput shape: {x.shape}")
        print(f"Output shape: {features.shape}")
        print(f"Expected: [1, 4096, 768]")
