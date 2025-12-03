#!/usr/bin/env python3
"""
Multi-scale crop ensemble for CLIP classification.
Test crops at different scales and ensemble predictions.

Expected: +2-5% improvement by capturing different context levels
"""

import os
import sys
import json
import torch
import numpy as np
from PIL import Image
from datetime import datetime
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'sam2'))

from dataset import BCSSDataset


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
    sys.stdout.flush()


class MultiScaleCLIPClassifier:
    """CLIP classifier with multi-scale crops."""
    
    def __init__(self, model_path=None, device='cuda'):
        self.device = device
        from transformers import CLIPProcessor, CLIPModel
        
        log("Loading CLIP...")
        if model_path and os.path.exists(model_path):
            self.model = CLIPModel.from_pretrained(model_path, local_files_only=True)
            self.processor = CLIPProcessor.from_pretrained(model_path, local_files_only=True)
        else:
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        self.model.to(device).eval()
        log("  CLIP loaded!")
        
        self.classes = ['tumor', 'stroma', 'lymphocyte', 'necrosis', 'blood_vessel']
        
        # Best prompts (fewshot)
        self.prompts = {
            'tumor': [
                "a histopathology image showing tumor cells",
                "breast cancer tumor tissue with irregular nuclei",
            ],
            'stroma': [
                "a histopathology image showing stromal tissue",
                "fibrous connective tissue stroma",
            ],
            'lymphocyte': [
                "a histopathology image showing lymphocytes",
                "immune cell infiltration with small dark cells",
            ],
            'necrosis': [
                "a histopathology image showing necrosis",
                "dead tissue with cellular debris",
            ],
            'blood_vessel': [
                "a histopathology image showing blood vessel",
                "vascular structure with red blood cells",
            ]
        }
        
        self._encode_prompts()
    
    def _encode_prompts(self):
        """Pre-encode text prompts."""
        self.text_features = {}
        with torch.no_grad():
            for cls, prompts in self.prompts.items():
                inputs = self.processor(text=prompts, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                embeds = self.model.get_text_features(**inputs)
                embeds = embeds / embeds.norm(dim=-1, keepdim=True)
                self.text_features[cls] = embeds.mean(dim=0)
        
        self.text_matrix = torch.stack([self.text_features[c] for c in self.classes])
        self.text_matrix = self.text_matrix / self.text_matrix.norm(dim=-1, keepdim=True)
    
    def get_multiscale_crops(self, image, mask, scales=[0.8, 1.0, 1.2, 1.5]):
        """Get crops at multiple scales around the mask."""
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any() or not cols.any():
            return []
        
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        h, w = mask.shape
        center_r = (rmin + rmax) // 2
        center_c = (cmin + cmax) // 2
        
        base_h = rmax - rmin
        base_w = cmax - cmin
        
        crops = []
        for scale in scales:
            new_h = int(base_h * scale)
            new_w = int(base_w * scale)
            
            # Calculate new bounds centered on the mask
            new_rmin = max(0, center_r - new_h // 2)
            new_rmax = min(h, center_r + new_h // 2)
            new_cmin = max(0, center_c - new_w // 2)
            new_cmax = min(w, center_c + new_w // 2)
            
            # Ensure minimum size
            if new_rmax - new_rmin < 10 or new_cmax - new_cmin < 10:
                continue
            
            crop = image.crop((new_cmin, new_rmin, new_cmax, new_rmax))
            crops.append(crop)
        
        return crops
    
    def classify(self, image: Image.Image, mask: np.ndarray = None) -> tuple:
        """Classify using single crop (baseline)."""
        if mask is not None:
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            if not rows.any() or not cols.any():
                return self.classes[0], 0.0, {}
            
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            pad = 10
            h, w = mask.shape
            rmin, rmax = max(0, rmin - pad), min(h, rmax + pad)
            cmin, cmax = max(0, cmin - pad), min(w, cmax + pad)
            image = image.crop((cmin, rmin, cmax, rmax))
        
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            embeds = self.model.get_image_features(**inputs)
            embeds = embeds / embeds.norm(dim=-1, keepdim=True)
            
            sims = (embeds @ self.text_matrix.T).squeeze()
            probs = torch.softmax(sims * 100, dim=-1)
            
            pred_idx = probs.argmax().item()
            confidence = probs[pred_idx].item()
            all_probs = {c: probs[i].item() for i, c in enumerate(self.classes)}
        
        return self.classes[pred_idx], confidence, all_probs
    
    def classify_multiscale(self, image: Image.Image, mask: np.ndarray, 
                           scales=[0.8, 1.0, 1.2, 1.5]) -> tuple:
        """Classify using multi-scale ensemble."""
        crops = self.get_multiscale_crops(image, mask, scales)
        
        if not crops:
            return self.classify(image, mask)
        
        all_probs = torch.zeros(len(self.classes), device=self.device)
        
        with torch.no_grad():
            for crop in crops:
                inputs = self.processor(images=crop, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                embeds = self.model.get_image_features(**inputs)
                embeds = embeds / embeds.norm(dim=-1, keepdim=True)
                
                sims = (embeds @ self.text_matrix.T).squeeze()
                probs = torch.softmax(sims * 100, dim=-1)
                all_probs += probs
        
        # Average
        all_probs = all_probs / len(crops)
        
        pred_idx = all_probs.argmax().item()
        confidence = all_probs[pred_idx].item()
        probs_dict = {c: all_probs[i].item() for i, c in enumerate(self.classes)}
        
        return self.classes[pred_idx], confidence, probs_dict


def main():
    log("=" * 70)
    log("MULTI-SCALE CLIP CLASSIFICATION")
    log("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log(f"Device: {device}")
    
    # Load dataset
    log("\nLoading dataset...")
    image_dir = os.path.join(project_root, 'data', 'bcss', 'images')
    mask_dir = os.path.join(project_root, 'data', 'bcss', 'masks')
    dataset = BCSSDataset(image_dir=image_dir, mask_dir=mask_dir, split='test')
    log(f"Loaded {len(dataset)} samples")
    
    # Load classifier
    clip_path = os.environ.get('CLIP_MODEL_PATH', None)
    classifier = MultiScaleCLIPClassifier(model_path=clip_path, device=device)
    
    class_map = {1: 'tumor', 2: 'stroma', 3: 'lymphocyte', 4: 'necrosis', 5: 'blood_vessel'}
    
    # Test configurations
    configs = {
        'single_scale': {'scales': [1.0]},
        'multi_scale_3': {'scales': [0.8, 1.0, 1.2]},
        'multi_scale_5': {'scales': [0.6, 0.8, 1.0, 1.2, 1.5]},
    }
    
    all_results = {}
    
    for config_name, config in configs.items():
        log(f"\n{'='*50}")
        log(f"Testing: {config_name}")
        log("=" * 50)
        
        results = {
            'correct': 0,
            'total': 0,
            'per_class': {c: {'correct': 0, 'total': 0} for c in classifier.classes}
        }
        
        pbar = tqdm(range(len(dataset)), desc=config_name, file=sys.stdout, ncols=80)
        
        for idx in pbar:
            sample = dataset[idx]
            image_np = sample.get('image_np')
            mask = sample['mask']
            
            if image_np.dtype != np.uint8:
                if image_np.max() <= 1:
                    image_np = (image_np * 255).astype(np.uint8)
                else:
                    image_np = image_np.astype(np.uint8)
            
            pil_image = Image.fromarray(image_np)
            
            if isinstance(mask, torch.Tensor):
                mask = mask.numpy()
            
            unique_classes = sample.get('unique_classes', [])
            if len(unique_classes) == 0:
                unique_classes = np.unique(mask)
                unique_classes = unique_classes[unique_classes > 0]
            
            for class_idx in unique_classes:
                class_name = class_map.get(int(class_idx))
                if class_name not in classifier.classes:
                    continue
                
                class_mask = (mask == class_idx)
                if not class_mask.any():
                    continue
                
                if len(config['scales']) == 1:
                    pred_class, conf, probs = classifier.classify(pil_image, class_mask)
                else:
                    pred_class, conf, probs = classifier.classify_multiscale(
                        pil_image, class_mask, config['scales']
                    )
                
                results['total'] += 1
                results['per_class'][class_name]['total'] += 1
                
                if pred_class == class_name:
                    results['correct'] += 1
                    results['per_class'][class_name]['correct'] += 1
            
            if results['total'] > 0:
                acc = results['correct'] / results['total'] * 100
                pbar.set_postfix({'acc': f'{acc:.1f}%'})
        
        overall_acc = results['correct'] / results['total'] if results['total'] > 0 else 0
        
        log(f"\n{config_name} Results: {overall_acc:.2%}")
        
        all_results[config_name] = {
            'accuracy': overall_acc,
            'scales': config['scales'],
            'per_class': {
                cls: {
                    'accuracy': results['per_class'][cls]['correct'] / results['per_class'][cls]['total']
                        if results['per_class'][cls]['total'] > 0 else 0
                }
                for cls in classifier.classes
            }
        }
    
    # Summary
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)
    
    for name, res in all_results.items():
        log(f"  {name:20s}: {res['accuracy']:.1%}")
    
    # Save results
    output_dir = os.path.join(project_root, 'results', 'multiscale_clip')
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    log(f"\nResults saved to: {output_dir}/")


if __name__ == '__main__':
    main()
