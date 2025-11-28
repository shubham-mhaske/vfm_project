"""
LoRA (Low-Rank Adaptation) adapter for SAM2.

This module implements LoRA adapters that can be injected into SAM2's transformer layers
to enable parameter-efficient fine-tuning. Based on the paper "LoRA: Low-Rank Adaptation 
of Large Language Models" (https://arxiv.org/abs/2106.09685) and inspired by Medical SAM 
Adapter (https://arxiv.org/abs/2304.12620).

Key Design Principles:
1. Freeze original SAM2 weights to preserve pretrained knowledge
2. Add small trainable LoRA matrices (A, B) to attention layers
3. Output = Original_Output + alpha * (B @ A @ x) where rank(A), rank(B) << hidden_dim
4. Typically adds only 0.1-1% new parameters

For BCSS dataset with only 85 training images, this approach helps prevent catastrophic
forgetting that occurs with full fine-tuning.
"""

import math
from typing import Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    A LoRA-enhanced Linear layer that adds low-rank decomposition to an existing linear layer.
    
    The forward pass computes:
        output = original_linear(x) + (alpha / r) * B @ A @ x
    
    where:
        - A: (in_features, r) - projects input to low-rank space
        - B: (r, out_features) - projects back to output space
        - alpha: scaling factor (typically same as r or fixed to a constant)
        - r: rank of the low-rank matrices (typically 4, 8, or 16)
    
    Args:
        original_linear: The original nn.Linear layer to wrap
        r: Rank of the LoRA matrices (default: 8)
        alpha: Scaling factor (default: same as r)
        dropout: Dropout rate for LoRA (default: 0.0)
    """
    
    def __init__(
        self,
        original_linear: nn.Linear,
        r: int = 8,
        alpha: float = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.original_linear = original_linear
        self.r = r
        self.alpha = alpha if alpha is not None else float(r)
        self.scaling = self.alpha / self.r
        
        in_features = original_linear.in_features
        out_features = original_linear.out_features
        
        # Freeze original weights
        for param in self.original_linear.parameters():
            param.requires_grad = False
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(in_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))
        
        # Initialize A with Kaiming and B with zeros (so LoRA starts as identity)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original forward
        original_output = self.original_linear(x)
        
        # LoRA forward: x @ A @ B * scaling
        lora_output = self.dropout(x) @ self.lora_A @ self.lora_B * self.scaling
        
        return original_output + lora_output
    
    def extra_repr(self) -> str:
        return f'in_features={self.original_linear.in_features}, out_features={self.original_linear.out_features}, r={self.r}, alpha={self.alpha}'


class LoRAMultiHeadAttention(nn.Module):
    """
    LoRA adapter for multi-head attention layers.
    
    Wraps the Q, K, V, and output projections with LoRA matrices.
    Can be configured to only apply LoRA to specific projections.
    
    Args:
        attention_module: The attention module containing qkv and proj layers
        r: Rank of LoRA matrices
        alpha: Scaling factor
        dropout: Dropout rate
        apply_to: Which projections to apply LoRA to ('qkv', 'q', 'k', 'v', 'out', or list)
    """
    
    def __init__(
        self,
        attention_module: nn.Module,
        r: int = 8,
        alpha: float = None,
        dropout: float = 0.0,
        apply_to: Union[str, List[str]] = 'qkv',
    ):
        super().__init__()
        
        self.attention_module = attention_module
        self.r = r
        self.alpha = alpha if alpha is not None else float(r)
        self.scaling = self.alpha / self.r
        
        # Determine which layers to apply LoRA to
        if isinstance(apply_to, str):
            apply_to = [apply_to]
        self.apply_to = apply_to
        
        # Find and wrap the appropriate linear layers
        self._wrap_layers(dropout)
    
    def _wrap_layers(self, dropout: float):
        """Find and wrap linear layers with LoRA."""
        # Common patterns in attention modules
        # SAM2 uses MultiScaleAttention with 'qkv' and 'proj' attributes
        
        if hasattr(self.attention_module, 'qkv') and 'qkv' in self.apply_to:
            original_qkv = self.attention_module.qkv
            if isinstance(original_qkv, nn.Linear):
                self.attention_module.qkv = LoRALinear(
                    original_qkv, self.r, self.alpha, dropout
                )
        
        if hasattr(self.attention_module, 'proj') and 'out' in self.apply_to:
            original_proj = self.attention_module.proj
            if isinstance(original_proj, nn.Linear):
                self.attention_module.proj = LoRALinear(
                    original_proj, self.r, self.alpha, dropout
                )
        
        # Also handle separate q, k, v projections (common in some transformers)
        for name in ['q_proj', 'k_proj', 'v_proj']:
            short_name = name[0]  # 'q', 'k', or 'v'
            if hasattr(self.attention_module, name) and short_name in self.apply_to:
                original = getattr(self.attention_module, name)
                if isinstance(original, nn.Linear):
                    setattr(self.attention_module, name, LoRALinear(
                        original, self.r, self.alpha, dropout
                    ))
    
    def forward(self, *args, **kwargs):
        return self.attention_module(*args, **kwargs)


class SAM2LoRAAdapter(nn.Module):
    """
    LoRA Adapter for SAM2 that wraps the entire model and adds LoRA to specified layers.
    
    This is the main class to use for fine-tuning SAM2 with LoRA. It:
    1. Freezes all original SAM2 parameters
    2. Adds LoRA matrices to attention layers in the image encoder and mask decoder
    3. Optionally keeps some layers fully trainable (e.g., final output heads)
    
    Args:
        sam2_model: The SAM2 model to adapt
        r: Rank of LoRA matrices (default: 8)
        alpha: Scaling factor (default: same as r)
        dropout: Dropout rate for LoRA (default: 0.0)
        target_modules: Which modules to apply LoRA to. Options:
            - 'image_encoder': Apply to Hiera image encoder
            - 'mask_decoder': Apply to SAM mask decoder transformer
            - 'memory_attention': Apply to memory attention module
            - 'all': Apply to all of the above
        trainable_modules: List of module name patterns to keep fully trainable
            (e.g., ['sam_mask_decoder.output_hypernetworks_mlps'])
    """
    
    def __init__(
        self,
        sam2_model: nn.Module,
        r: int = 8,
        alpha: float = None,
        dropout: float = 0.0,
        target_modules: Union[str, List[str]] = 'all',
        trainable_modules: List[str] = None,
    ):
        super().__init__()
        
        self.sam2_model = sam2_model
        self.r = r
        self.alpha = alpha if alpha is not None else float(r)
        self.dropout = dropout
        
        if isinstance(target_modules, str):
            target_modules = [target_modules] if target_modules != 'all' else [
                'image_encoder', 'mask_decoder', 'memory_attention'
            ]
        self.target_modules = target_modules
        self.trainable_modules = trainable_modules or []
        
        # First, freeze all parameters
        self._freeze_all_params()
        
        # Then add LoRA adapters
        self.lora_layers = nn.ModuleDict()
        self._add_lora_adapters()
        
        # Unfreeze any specified trainable modules
        self._unfreeze_trainable_modules()
        
        # Count and report parameters
        self._report_param_stats()
    
    def _freeze_all_params(self):
        """Freeze all parameters in the original model."""
        for param in self.sam2_model.parameters():
            param.requires_grad = False
    
    def _add_lora_adapters(self):
        """Add LoRA adapters to target modules."""
        lora_count = 0
        
        if 'image_encoder' in self.target_modules:
            lora_count += self._add_lora_to_image_encoder()
        
        if 'mask_decoder' in self.target_modules:
            lora_count += self._add_lora_to_mask_decoder()
        
        if 'memory_attention' in self.target_modules:
            lora_count += self._add_lora_to_memory_attention()
        
        print(f"[LoRA] Added {lora_count} LoRA layers with rank={self.r}")
    
    def _add_lora_to_image_encoder(self) -> int:
        """Add LoRA to the Hiera image encoder attention layers."""
        count = 0
        image_encoder = self.sam2_model.image_encoder
        
        # Hiera uses MultiScaleBlock with MultiScaleAttention
        for name, module in image_encoder.named_modules():
            if hasattr(module, 'attn') and hasattr(module.attn, 'qkv'):
                # This is a MultiScaleBlock with attention
                attn = module.attn
                if isinstance(attn.qkv, nn.Linear):
                    attn.qkv = LoRALinear(attn.qkv, self.r, self.alpha, self.dropout)
                    count += 1
                if isinstance(attn.proj, nn.Linear):
                    attn.proj = LoRALinear(attn.proj, self.r, self.alpha, self.dropout)
                    count += 1
        
        print(f"[LoRA] Image encoder: {count} layers")
        return count
    
    def _add_lora_to_mask_decoder(self) -> int:
        """Add LoRA to the SAM mask decoder transformer."""
        count = 0
        mask_decoder = self.sam2_model.sam_mask_decoder
        
        # The mask decoder has a TwoWayTransformer with TwoWayAttentionBlocks
        if hasattr(mask_decoder, 'transformer'):
            transformer = mask_decoder.transformer
            for layer in transformer.layers:
                # Each TwoWayAttentionBlock has self_attn and cross_attn modules
                for attn_name in ['self_attn', 'cross_attn_token_to_image', 'cross_attn_image_to_token']:
                    if hasattr(layer, attn_name):
                        attn = getattr(layer, attn_name)
                        # These are Attention modules with q_proj, k_proj, v_proj, out_proj
                        for proj_name in ['q_proj', 'k_proj', 'v_proj', 'out_proj']:
                            if hasattr(attn, proj_name):
                                proj = getattr(attn, proj_name)
                                if isinstance(proj, nn.Linear):
                                    setattr(attn, proj_name, LoRALinear(
                                        proj, self.r, self.alpha, self.dropout
                                    ))
                                    count += 1
        
        print(f"[LoRA] Mask decoder: {count} layers")
        return count
    
    def _add_lora_to_memory_attention(self) -> int:
        """Add LoRA to the memory attention module."""
        count = 0
        if hasattr(self.sam2_model, 'memory_attention'):
            memory_attention = self.sam2_model.memory_attention
            for name, module in memory_attention.named_modules():
                if isinstance(module, nn.Linear) and 'attn' in name.lower():
                    # Replace with LoRA version
                    # This is trickier as we need to find the parent module
                    pass  # Skip for now - memory attention structure varies
        
        print(f"[LoRA] Memory attention: {count} layers")
        return count
    
    def _unfreeze_trainable_modules(self):
        """Unfreeze specified modules to be fully trainable."""
        for pattern in self.trainable_modules:
            for name, param in self.sam2_model.named_parameters():
                if pattern in name:
                    param.requires_grad = True
    
    def _report_param_stats(self):
        """Report parameter statistics."""
        total_params = sum(p.numel() for p in self.sam2_model.parameters())
        trainable_params = sum(p.numel() for p in self.sam2_model.parameters() if p.requires_grad)
        
        print(f"[LoRA] Total parameters: {total_params:,}")
        print(f"[LoRA] Trainable parameters: {trainable_params:,}")
        print(f"[LoRA] Trainable percentage: {100 * trainable_params / total_params:.2f}%")
    
    def forward(self, *args, **kwargs):
        """Forward pass through the adapted SAM2 model."""
        return self.sam2_model(*args, **kwargs)
    
    def get_trainable_params(self) -> List[nn.Parameter]:
        """Get list of trainable parameters for optimizer."""
        return [p for p in self.sam2_model.parameters() if p.requires_grad]
    
    def save_lora_weights(self, path: str):
        """Save only the LoRA weights (for efficient storage)."""
        lora_state_dict = {}
        for name, param in self.sam2_model.named_parameters():
            if param.requires_grad:
                lora_state_dict[name] = param.data
        torch.save(lora_state_dict, path)
        print(f"[LoRA] Saved {len(lora_state_dict)} trainable parameter tensors to {path}")
    
    def load_lora_weights(self, path: str):
        """Load LoRA weights."""
        lora_state_dict = torch.load(path, map_location='cpu')
        current_state = self.sam2_model.state_dict()
        current_state.update(lora_state_dict)
        self.sam2_model.load_state_dict(current_state)
        print(f"[LoRA] Loaded {len(lora_state_dict)} LoRA weight tensors from {path}")


def apply_lora_to_sam2(
    sam2_model: nn.Module,
    r: int = 8,
    alpha: float = None,
    dropout: float = 0.0,
    target_modules: Union[str, List[str]] = 'image_encoder',
    trainable_output_head: bool = True,
) -> SAM2LoRAAdapter:
    """
    Convenience function to apply LoRA to a SAM2 model.
    
    Args:
        sam2_model: The SAM2 model to adapt
        r: Rank of LoRA matrices (lower = fewer params, higher = more capacity)
            - r=4: ~0.1% trainable params
            - r=8: ~0.2% trainable params  
            - r=16: ~0.4% trainable params
        alpha: Scaling factor (default: same as r)
        dropout: Dropout rate for LoRA regularization
        target_modules: Where to apply LoRA ('image_encoder', 'mask_decoder', 'all')
        trainable_output_head: Whether to keep output MLP heads trainable
    
    Returns:
        SAM2LoRAAdapter wrapping the model
    
    Example:
        >>> from sam2.build_sam import build_sam2
        >>> sam2_model = build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml", checkpoint)
        >>> lora_model = apply_lora_to_sam2(sam2_model, r=8, target_modules='image_encoder')
        >>> optimizer = torch.optim.AdamW(lora_model.get_trainable_params(), lr=1e-4)
    """
    trainable_modules = []
    if trainable_output_head:
        # Keep the final output heads trainable
        trainable_modules.extend([
            'sam_mask_decoder.output_hypernetworks_mlps',
            'sam_mask_decoder.iou_prediction_head',
        ])
    
    return SAM2LoRAAdapter(
        sam2_model=sam2_model,
        r=r,
        alpha=alpha,
        dropout=dropout,
        target_modules=target_modules,
        trainable_modules=trainable_modules,
    )


# Example usage and testing
if __name__ == "__main__":
    # Test LoRALinear
    print("Testing LoRALinear...")
    original = nn.Linear(512, 256)
    lora_linear = LoRALinear(original, r=8)
    
    x = torch.randn(2, 512)
    out = lora_linear(x)
    print(f"  Input: {x.shape}, Output: {out.shape}")
    
    # Count parameters
    orig_params = sum(p.numel() for p in original.parameters())
    lora_params = sum(p.numel() for p in lora_linear.parameters() if p.requires_grad)
    print(f"  Original params: {orig_params}, LoRA trainable params: {lora_params}")
    print(f"  LoRA overhead: {100 * lora_params / orig_params:.2f}%")
    print("  LoRALinear test passed!")
