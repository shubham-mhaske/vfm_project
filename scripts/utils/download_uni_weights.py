"""
Download UNI Encoder Weights
============================

UNI is a general-purpose self-supervised learning model for computational pathology,
trained on >100 million H&E histopathology patches.

Paper: "A General-Purpose Self-Supervised Model for Computational Pathology"
Repository: https://huggingface.co/MahmoodLab/UNI

The model requires authentication with HuggingFace Hub.
You need to:
1. Create a HuggingFace account: https://huggingface.co/join
2. Accept the UNI model terms: https://huggingface.co/MahmoodLab/UNI
3. Generate an access token: https://huggingface.co/settings/tokens
4. Login via CLI: huggingface-cli login

Usage:
    python src/download_uni_weights.py
    
Or manually:
    pip install huggingface_hub
    huggingface-cli login
    python -c "from huggingface_hub import snapshot_download; snapshot_download('MahmoodLab/UNI', local_dir='models/uni')"
"""

import os
import sys

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)


def download_uni_weights():
    """Download UNI weights from HuggingFace Hub."""
    
    output_dir = os.path.join(project_root, 'models', 'uni')
    checkpoint_path = os.path.join(output_dir, 'pytorch_model.bin')
    
    print("="*60)
    print("UNI Encoder Weight Download")
    print("="*60)
    
    # Check if already exists
    if os.path.exists(checkpoint_path):
        size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
        print(f"✓ UNI weights already exist at: {checkpoint_path}")
        print(f"  Size: {size_mb:.1f} MB")
        return checkpoint_path
    
    print("\nUNI Model Information:")
    print("  - Model: ViT-Large/16 (DINOv2-style)")
    print("  - Training: >100M histopathology patches")
    print("  - Input: 224x224 H&E patches")
    print("  - Output: 1024-dim embeddings")
    print("  - Size: ~1.3 GB")
    
    print("\n" + "-"*40)
    print("IMPORTANT: Before downloading, you must:")
    print("  1. Accept the model terms at:")
    print("     https://huggingface.co/MahmoodLab/UNI")
    print("  2. Login to HuggingFace CLI:")
    print("     huggingface-cli login")
    print("-"*40 + "\n")
    
    try:
        from huggingface_hub import snapshot_download, HfApi
        from huggingface_hub.utils import HfHubHTTPError
    except ImportError:
        print("ERROR: huggingface_hub not installed.")
        print("Install with: pip install huggingface_hub")
        return None
    
    # Check login status
    try:
        api = HfApi()
        user = api.whoami()
        print(f"✓ Logged in as: {user['name']}")
    except Exception as e:
        print("ERROR: Not logged in to HuggingFace.")
        print("Run: huggingface-cli login")
        return None
    
    # Download
    print(f"\nDownloading UNI weights to: {output_dir}")
    print("This may take several minutes...\n")
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        snapshot_download(
            repo_id="MahmoodLab/UNI",
            local_dir=output_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        
        print(f"\n✓ Download complete!")
        
        # Verify checkpoint exists
        if os.path.exists(checkpoint_path):
            size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
            print(f"✓ Checkpoint saved: {checkpoint_path}")
            print(f"  Size: {size_mb:.1f} MB")
            return checkpoint_path
        else:
            # Try to find the checkpoint with different name
            for f in os.listdir(output_dir):
                if f.endswith('.bin') or f.endswith('.pt'):
                    found_path = os.path.join(output_dir, f)
                    print(f"✓ Checkpoint found: {found_path}")
                    return found_path
            
            print("WARNING: Checkpoint not found in expected location.")
            print(f"Contents of {output_dir}:")
            for f in os.listdir(output_dir):
                print(f"  - {f}")
            return None
            
    except HfHubHTTPError as e:
        if "401" in str(e) or "403" in str(e):
            print("ERROR: Access denied. Make sure you have:")
            print("  1. Accepted terms at https://huggingface.co/MahmoodLab/UNI")
            print("  2. Valid HuggingFace token")
        else:
            print(f"ERROR downloading: {e}")
        return None
    except Exception as e:
        print(f"ERROR: {e}")
        return None


def verify_uni_encoder():
    """Test that UNI encoder can be loaded."""
    checkpoint_path = os.path.join(project_root, 'models', 'uni', 'pytorch_model.bin')
    
    if not os.path.exists(checkpoint_path):
        print("\n✗ Cannot verify - checkpoint not found")
        return False
    
    print("\n" + "-"*40)
    print("Verifying UNI encoder...")
    
    try:
        import torch
        
        # Try loading weights
        print("  Loading checkpoint...")
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        
        # Check structure
        num_params = len(state_dict)
        total_params = sum(p.numel() for p in state_dict.values())
        
        print(f"  ✓ Loaded {num_params} tensors")
        print(f"  ✓ Total parameters: {total_params:,}")
        
        # Try creating encoder
        print("  Creating UNI encoder...")
        from src.uni_encoder import UNIEncoder
        
        encoder = UNIEncoder(checkpoint_path=checkpoint_path, freeze=True)
        print(f"  ✓ UNI encoder created successfully")
        
        # Test forward pass
        print("  Testing forward pass...")
        test_input = torch.randn(1, 3, 1024, 1024)
        with torch.no_grad():
            output = encoder(test_input)
        
        print(f"  ✓ Input shape: {list(test_input.shape)}")
        print(f"  ✓ Output shape: {list(output.shape)}")
        print(f"  ✓ Expected: [1, 4096, 1024] (64x64 patches, 1024-dim)")
        
        print("\n✓ UNI encoder verification PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download UNI encoder weights")
    parser.add_argument('--verify-only', action='store_true', 
                       help="Only verify existing weights, don't download")
    args = parser.parse_args()
    
    if args.verify_only:
        verify_uni_encoder()
    else:
        result = download_uni_weights()
        if result:
            verify_uni_encoder()
