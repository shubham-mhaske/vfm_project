#!/usr/bin/env python3
"""
Test PLIP (Pathology Language-Image Pretraining) for tissue classification.
PLIP is trained on 200K pathology image-text pairs - should significantly
outperform general CLIP on histopathology.

Expected improvement: 38.9% (CLIP) → 55-65% (PLIP)
"""

import os
import sys
import json
import torch
import numpy as np
from PIL import Image
from datetime import datetime
from tqdm import tqdm

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'sam2'))

from dataset import BCSSDataset


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


class PLIPClassifier:
    """PLIP-based classifier for pathology images."""
    
    def __init__(self, device='cuda', model_path=None):
        self.device = device
        
        # Try to import and load PLIP
        try:
            from transformers import CLIPProcessor, CLIPModel
            
            log("  Importing transformers library...")
            sys.stdout.flush()
            
            # Check for local model path (for HPRC offline use)
            if model_path is None:
                model_path = os.environ.get('PLIP_MODEL_PATH', None)
            
            if model_path and os.path.exists(model_path):
                log(f"  Loading PLIP from local path: {model_path}")
                sys.stdout.flush()
                self.model = CLIPModel.from_pretrained(model_path, local_files_only=True)
                self.processor = CLIPProcessor.from_pretrained(model_path, local_files_only=True)
            else:
                log("  Downloading/loading PLIP model (vinid/plip)...")
                log("  (This downloads ~600MB on first run)")
                sys.stdout.flush()
                self.model = CLIPModel.from_pretrained("vinid/plip")
                self.processor = CLIPProcessor.from_pretrained("vinid/plip")
            
            log("  Model weights loaded!")
            sys.stdout.flush()
            
            log("  Processor loaded!")
            sys.stdout.flush()
            
            log(f"  Moving model to {device}...")
            sys.stdout.flush()
            self.model.to(device)
            self.model.eval()
            
            log("  PLIP loaded successfully!")
            sys.stdout.flush()
            self.use_transformers = True
            
        except Exception as e:
            log(f"Error loading PLIP with transformers: {e}")
            log("Trying open_clip...")
            
            try:
                import open_clip
                # Alternative: use open_clip if available
                self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                    'ViT-B-32', pretrained='openai'
                )
                self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
                self.model.to(device)
                self.model.eval()
                self.use_transformers = False
                log("Loaded OpenCLIP as fallback")
            except Exception as e2:
                raise RuntimeError(f"Could not load PLIP or OpenCLIP: {e2}")
        
        # Pathology-specific prompts (optimized for PLIP)
        self.class_prompts = {
            'tumor': [
                "tumor cells in breast tissue",
                "invasive carcinoma cells with irregular nuclei",
                "malignant epithelial cells forming glands",
                "breast cancer tumor region",
            ],
            'stroma': [
                "stromal tissue with fibroblasts",
                "connective tissue stroma",
                "fibrous stromal region in breast",
                "dense collagenous stroma",
            ],
            'lymphocyte': [
                "lymphocyte infiltration",
                "tumor infiltrating lymphocytes",
                "immune cell aggregates",
                "small round lymphocytes with dark nuclei",
            ],
            'necrosis': [
                "necrotic tissue region",
                "tissue necrosis with cell death",
                "dead cells and debris",
                "necrotic tumor area",
            ],
            'blood_vessel': [
                "blood vessel cross section",
                "vascular structure with red blood cells",
                "endothelial lined vessel",
                "capillary or small blood vessel",
            ]
        }
        
        self.classes = list(self.class_prompts.keys())
        log("  Encoding text prompts...")
        sys.stdout.flush()
        self._encode_text_prompts()
        log("  Text prompts encoded!")
        sys.stdout.flush()
    
    def _encode_text_prompts(self):
        """Pre-encode all text prompts."""
        self.text_features = {}
        
        with torch.no_grad():
            for class_name, prompts in self.class_prompts.items():
                if self.use_transformers:
                    inputs = self.processor(text=prompts, return_tensors="pt", padding=True)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    text_embeds = self.model.get_text_features(**inputs)
                else:
                    tokens = self.tokenizer(prompts).to(self.device)
                    text_embeds = self.model.encode_text(tokens)
                
                # Average embeddings for this class
                text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
                self.text_features[class_name] = text_embeds.mean(dim=0)
        
        # Stack into matrix
        self.text_matrix = torch.stack([self.text_features[c] for c in self.classes])
        self.text_matrix = self.text_matrix / self.text_matrix.norm(dim=-1, keepdim=True)
    
    def classify(self, image: Image.Image, mask: np.ndarray = None) -> tuple:
        """
        Classify an image region.
        
        Args:
            image: PIL Image
            mask: Optional binary mask to crop region
            
        Returns:
            (predicted_class, confidence, all_probs)
        """
        # Crop to mask if provided
        if mask is not None:
            # Get bounding box
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            if not rows.any() or not cols.any():
                return self.classes[0], 0.0, {c: 0.0 for c in self.classes}
            
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            
            # Add padding
            h, w = mask.shape
            pad = 10
            rmin, rmax = max(0, rmin - pad), min(h, rmax + pad)
            cmin, cmax = max(0, cmin - pad), min(w, cmax + pad)
            
            # Crop image
            image = image.crop((cmin, rmin, cmax, rmax))
        
        with torch.no_grad():
            if self.use_transformers:
                inputs = self.processor(images=image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                image_embeds = self.model.get_image_features(**inputs)
            else:
                img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
                image_embeds = self.model.encode_image(img_tensor)
            
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            
            # Compute similarities
            similarities = (image_embeds @ self.text_matrix.T).squeeze()
            probs = torch.softmax(similarities * 100, dim=-1)
            
            pred_idx = probs.argmax().item()
            confidence = probs[pred_idx].item()
            
            all_probs = {c: probs[i].item() for i, c in enumerate(self.classes)}
        
        return self.classes[pred_idx], confidence, all_probs


def main():
    log("=" * 70)
    log("PLIP TEST: Pathology-Specific CLIP for Tissue Classification")
    log("=" * 70)
    sys.stdout.flush()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log(f"Device: {device}")
    if device == 'cuda':
        log(f"GPU: {torch.cuda.get_device_name(0)}")
        log(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    sys.stdout.flush()
    
    # Load dataset
    log("Loading dataset...")
    sys.stdout.flush()
    image_dir = os.path.join(project_root, 'data', 'bcss', 'images')
    mask_dir = os.path.join(project_root, 'data', 'bcss', 'masks')
    log(f"  Image dir: {image_dir}")
    log(f"  Mask dir: {mask_dir}")
    sys.stdout.flush()
    dataset = BCSSDataset(image_dir=image_dir, mask_dir=mask_dir, split='test')
    log(f"Loaded {len(dataset)} samples")
    sys.stdout.flush()
    
    # Load PLIP
    log("\n" + "-" * 50)
    log("Loading PLIP model...")
    log("  This may take a few minutes on first run (downloading ~600MB)")
    sys.stdout.flush()
    try:
        classifier = PLIPClassifier(device=device)
    except Exception as e:
        log(f"Failed to load PLIP: {e}")
        log("\nTo install PLIP, run:")
        log("  pip install transformers")
        log("\nOr for open_clip:")
        log("  pip install open_clip_torch")
        sys.stdout.flush()
        return
    
    log("Text prompts encoded!")
    sys.stdout.flush()
    
    # Evaluate
    log("\n" + "-" * 50)
    log("Running evaluation...")
    log(f"  Processing {len(dataset)} images...")
    sys.stdout.flush()
    
    results = {
        'correct': 0,
        'total': 0,
        'per_class': {c: {'correct': 0, 'total': 0} for c in classifier.classes}
    }
    
    # Use tqdm with flush
    pbar = tqdm(range(len(dataset)), desc="Evaluating", file=sys.stdout, 
                dynamic_ncols=False, ncols=80)
    
    for idx in pbar:
        sample = dataset[idx]
        image = sample['image']
        mask = sample['mask']
        filename = sample.get('filename', f'sample_{idx}')
        
        # Convert image to PIL if needed
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        if isinstance(image, np.ndarray):
            if image.max() <= 1:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        
        # Get unique classes in this image
        unique_classes = sample.get('unique_classes', [])
        if not unique_classes:
            unique_classes = np.unique(mask)
            unique_classes = unique_classes[unique_classes > 0]
        
        # Class mapping
        class_map = {1: 'tumor', 2: 'stroma', 3: 'lymphocyte', 4: 'necrosis', 5: 'blood_vessel'}
        
        img_correct = 0
        img_total = 0
        
        for class_idx in unique_classes:
            if isinstance(class_idx, str):
                class_name = class_idx
            else:
                class_name = class_map.get(int(class_idx))
            
            if class_name not in classifier.classes:
                continue
            
            # Get mask for this class
            if isinstance(class_idx, str):
                class_mask = (mask == list(class_map.values()).index(class_idx) + 1)
            else:
                class_mask = (mask == class_idx)
            
            if not class_mask.any():
                continue
            
            # Classify
            pred_class, conf, probs = classifier.classify(image, class_mask)
            
            # Record result
            results['total'] += 1
            results['per_class'][class_name]['total'] += 1
            img_total += 1
            
            if pred_class == class_name:
                results['correct'] += 1
                results['per_class'][class_name]['correct'] += 1
                img_correct += 1
        
        # Update progress bar with running accuracy
        if results['total'] > 0:
            running_acc = results['correct'] / results['total'] * 100
            pbar.set_postfix({'acc': f'{running_acc:.1f}%', 'n': results['total']})
        
        # Periodic logging every 10 images
        if (idx + 1) % 10 == 0:
            sys.stdout.flush()
    
    # Calculate metrics
    overall_acc = results['correct'] / results['total'] if results['total'] > 0 else 0
    
    log("\n" + "=" * 70)
    log("RESULTS")
    log("=" * 70)
    
    log(f"\nOverall Accuracy: {overall_acc:.2%} ({results['correct']}/{results['total']})")
    log(f"\nComparison to CLIP: 38.9% → {overall_acc:.1%}")
    
    log("\nPer-Class Breakdown:")
    log("-" * 50)
    for class_name in classifier.classes:
        c = results['per_class'][class_name]
        acc = c['correct'] / c['total'] if c['total'] > 0 else 0
        log(f"  {class_name:15s}: {acc:6.2%} ({c['correct']:3d}/{c['total']:3d})")
    
    # Save results
    output_dir = os.path.join(project_root, 'results', 'plip_test')
    os.makedirs(output_dir, exist_ok=True)
    
    output = {
        'model': 'PLIP (vinid/plip)',
        'overall_accuracy': overall_acc,
        'total_samples': results['total'],
        'correct': results['correct'],
        'per_class': {
            c: {
                'accuracy': results['per_class'][c]['correct'] / results['per_class'][c]['total'] 
                    if results['per_class'][c]['total'] > 0 else 0,
                'correct': results['per_class'][c]['correct'],
                'total': results['per_class'][c]['total']
            }
            for c in classifier.classes
        },
        'comparison': {
            'clip_baseline': 0.389,
            'improvement': overall_acc - 0.389
        },
        'timestamp': datetime.now().isoformat()
    }
    
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(output, f, indent=2)
    
    log(f"\nResults saved to: {output_dir}/metrics.json")


if __name__ == '__main__':
    main()
