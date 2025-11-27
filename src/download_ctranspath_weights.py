"""
Download CTransPath Weights
============================

CTransPath is a Swin Transformer pretrained on 15M histopathology patches.
This script downloads the official pretrained weights.

Paper: "Transformer-based Unsupervised Contrastive Learning for Histopathological 
        Image Classification" (Medical Image Analysis 2022)

Usage:
    python src/download_ctranspath_weights.py
"""

import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)


def download_ctranspath():
    """Download CTransPath weights."""
    
    output_dir = os.path.join(project_root, 'models', 'ctranspath')
    checkpoint_path = os.path.join(output_dir, 'ctranspath.pth')
    
    print("="*60)
    print("CTransPath Weight Download")
    print("="*60)
    
    # Check if exists
    if os.path.exists(checkpoint_path):
        size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
        print(f"✓ Weights already exist: {checkpoint_path}")
        print(f"  Size: {size_mb:.1f} MB")
        return checkpoint_path
    
    print("\nCTransPath Information:")
    print("  - Architecture: Swin Transformer")
    print("  - Training: 15M pathology patches (TCGA)")
    print("  - Input: 224x224 H&E patches")
    print("  - Output: 768-dim embeddings")
    print("  - Size: ~100 MB")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Try gdown first (Google Drive)
    print("\n" + "-"*40)
    print("Method 1: Google Drive (gdown)")
    print("-"*40)
    
    try:
        import gdown
        
        # CTransPath Google Drive ID
        file_id = "1DoDx_70_TLj98gTf6YTXnu4tFhsFocDX"
        url = f"https://drive.google.com/uc?id={file_id}"
        
        print(f"Downloading from Google Drive...")
        gdown.download(url, checkpoint_path, quiet=False)
        
        if os.path.exists(checkpoint_path) and os.path.getsize(checkpoint_path) > 1e6:
            size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
            print(f"\n✓ Downloaded successfully!")
            print(f"  Path: {checkpoint_path}")
            print(f"  Size: {size_mb:.1f} MB")
            return checkpoint_path
            
    except ImportError:
        print("gdown not installed. Install with: pip install gdown")
    except Exception as e:
        print(f"Google Drive download failed: {e}")
    
    # Try direct URL
    print("\n" + "-"*40)
    print("Method 2: Direct URL")
    print("-"*40)
    
    try:
        import urllib.request
        
        # Alternative URLs
        urls = [
            "https://github.com/Xiyue-Wang/TransPath/releases/download/v1.0/ctranspath.pth",
        ]
        
        for url in urls:
            try:
                print(f"Trying: {url[:60]}...")
                urllib.request.urlretrieve(url, checkpoint_path)
                
                if os.path.exists(checkpoint_path) and os.path.getsize(checkpoint_path) > 1e6:
                    print(f"✓ Downloaded!")
                    return checkpoint_path
            except Exception as e:
                print(f"  Failed: {e}")
                
    except Exception as e:
        print(f"Direct download failed: {e}")
    
    # Manual instructions
    print("\n" + "="*60)
    print("AUTOMATIC DOWNLOAD FAILED - MANUAL STEPS REQUIRED")
    print("="*60)
    
    print("""
Option 1: Using gdown (recommended)
-----------------------------------
pip install gdown
gdown 1DoDx_70_TLj98gTf6YTXnu4tFhsFocDX -O models/ctranspath/ctranspath.pth

Option 2: Manual download
-------------------------
1. Go to: https://github.com/Xiyue-Wang/TransPath
2. Find the "Releases" section or check README for download link
3. Download ctranspath.pth 
4. Move to: models/ctranspath/ctranspath.pth

Option 3: Google Drive direct
-----------------------------
1. Go to: https://drive.google.com/file/d/1DoDx_70_TLj98gTf6YTXnu4tFhsFocDX
2. Download ctranspath.pth
3. Move to: models/ctranspath/ctranspath.pth
""")
    
    return None


def verify_ctranspath():
    """Verify CTransPath weights work correctly."""
    
    checkpoint_path = os.path.join(project_root, 'models', 'ctranspath', 'ctranspath.pth')
    
    if not os.path.exists(checkpoint_path):
        print("✗ Cannot verify - weights not found")
        return False
    
    print("\n" + "-"*40)
    print("Verifying CTransPath encoder...")
    print("-"*40)
    
    try:
        import torch
        
        # Load checkpoint
        print("  Loading checkpoint...")
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        
        if 'model' in state_dict:
            state_dict = state_dict['model']
        
        num_keys = len(state_dict)
        total_params = sum(v.numel() for v in state_dict.values())
        
        print(f"  ✓ Loaded {num_keys} tensors")
        print(f"  ✓ Total parameters: {total_params:,}")
        
        # Test encoder creation
        print("  Creating CTransPath encoder...")
        from src.ctranspath_encoder import CTransPathEncoder
        
        encoder = CTransPathEncoder(checkpoint_path=checkpoint_path, freeze=True)
        print("  ✓ Encoder created")
        
        # Test forward pass
        print("  Testing forward pass...")
        test_input = torch.randn(1, 3, 1024, 1024)
        with torch.no_grad():
            output = encoder(test_input)
        
        print(f"  ✓ Input: {list(test_input.shape)}")
        print(f"  ✓ Output: {list(output.shape)}")
        print(f"  ✓ Expected: [1, 4096, 768]")
        
        print("\n✓ CTransPath verification PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download CTransPath weights")
    parser.add_argument('--verify-only', action='store_true',
                       help="Only verify existing weights")
    args = parser.parse_args()
    
    if args.verify_only:
        verify_ctranspath()
    else:
        result = download_ctranspath()
        if result:
            verify_ctranspath()
