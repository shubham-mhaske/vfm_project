#!/usr/bin/env python3
"""
Ensemble CLIP + PLIP for classification.
Use CLIP for tumor/stroma/blood_vessel, PLIP for lymphocyte/necrosis.

Expected: ~45-50% (combining strengths of both models)
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


class EnsembleClassifier:
    """Ensemble CLIP + PLIP classifier."""
    
    def __init__(self, clip_path=None, plip_path=None, device='cuda'):
        self.device = device
        from transformers import CLIPProcessor, CLIPModel
        
        # Load CLIP
        log("Loading CLIP...")
        if clip_path and os.path.exists(clip_path):
            self.clip_model = CLIPModel.from_pretrained(clip_path, local_files_only=True)
            self.clip_processor = CLIPProcessor.from_pretrained(clip_path, local_files_only=True)
        else:
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model.to(device).eval()
        log("  CLIP loaded!")
        
        # Load PLIP
        log("Loading PLIP...")
        if plip_path and os.path.exists(plip_path):
            self.plip_model = CLIPModel.from_pretrained(plip_path, local_files_only=True)
            self.plip_processor = CLIPProcessor.from_pretrained(plip_path, local_files_only=True)
        else:
            self.plip_model = CLIPModel.from_pretrained("vinid/plip")
            self.plip_processor = CLIPProcessor.from_pretrained("vinid/plip")
        self.plip_model.to(device).eval()
        log("  PLIP loaded!")
        
        self.classes = ['tumor', 'stroma', 'lymphocyte', 'necrosis', 'blood_vessel']
        
        # Which model to use for each class (based on experimental results)
        # CLIP better: tumor (60%), stroma (64%), blood_vessel (23%)
        # PLIP better: lymphocyte (16% vs 2.7%), necrosis (48% vs 8.7%)
        self.model_assignment = {
            'tumor': 'clip',
            'stroma': 'clip',
            'lymphocyte': 'plip',
            'necrosis': 'plip',
            'blood_vessel': 'clip'
        }
        
        # Prompts optimized for each model
        self.clip_prompts = {
            'tumor': ["a histopathology image showing tumor cells", "breast cancer tumor tissue"],
            'stroma': ["a histopathology image showing stromal tissue", "connective tissue stroma"],
            'lymphocyte': ["a histopathology image showing lymphocytes", "immune cell infiltration"],
            'necrosis': ["a histopathology image showing necrosis", "dead tissue region"],
            'blood_vessel': ["a histopathology image showing blood vessel", "vascular structure"]
        }
        
        self.plip_prompts = {
            'tumor': ["tumor cells in breast tissue", "invasive carcinoma cells"],
            'stroma': ["stromal tissue with fibroblasts", "connective tissue stroma"],
            'lymphocyte': ["lymphocyte infiltration", "tumor infiltrating lymphocytes"],
            'necrosis': ["necrotic tissue region", "tissue necrosis with cell death"],
            'blood_vessel': ["blood vessel cross section", "vascular structure"]
        }
        
        self._encode_prompts()
    
    def _encode_prompts(self):
        """Pre-encode text prompts for both models."""
        log("Encoding text prompts...")
        
        # CLIP prompts
        self.clip_text_features = {}
        with torch.no_grad():
            for cls, prompts in self.clip_prompts.items():
                inputs = self.clip_processor(text=prompts, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                embeds = self.clip_model.get_text_features(**inputs)
                embeds = embeds / embeds.norm(dim=-1, keepdim=True)
                self.clip_text_features[cls] = embeds.mean(dim=0)
        
        self.clip_text_matrix = torch.stack([self.clip_text_features[c] for c in self.classes])
        self.clip_text_matrix = self.clip_text_matrix / self.clip_text_matrix.norm(dim=-1, keepdim=True)
        
        # PLIP prompts
        self.plip_text_features = {}
        with torch.no_grad():
            for cls, prompts in self.plip_prompts.items():
                inputs = self.plip_processor(text=prompts, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                embeds = self.plip_model.get_text_features(**inputs)
                embeds = embeds / embeds.norm(dim=-1, keepdim=True)
                self.plip_text_features[cls] = embeds.mean(dim=0)
        
        self.plip_text_matrix = torch.stack([self.plip_text_features[c] for c in self.classes])
        self.plip_text_matrix = self.plip_text_matrix / self.plip_text_matrix.norm(dim=-1, keepdim=True)
        
        log("  Prompts encoded!")
    
    def _crop_to_mask(self, image, mask):
        """Crop image to mask bounding box."""
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any() or not cols.any():
            return None
        
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        h, w = mask.shape
        pad = 10
        rmin, rmax = max(0, rmin - pad), min(h, rmax + pad)
        cmin, cmax = max(0, cmin - pad), min(w, cmax + pad)
        
        return image.crop((cmin, rmin, cmax, rmax))
    
    def classify(self, image: Image.Image, mask: np.ndarray = None) -> tuple:
        """Classify using ensemble (weighted by model assignment)."""
        if mask is not None:
            image = self._crop_to_mask(image, mask)
            if image is None:
                return self.classes[0], 0.0, {}
        
        with torch.no_grad():
            # Get CLIP predictions
            clip_inputs = self.clip_processor(images=image, return_tensors="pt")
            clip_inputs = {k: v.to(self.device) for k, v in clip_inputs.items()}
            clip_embeds = self.clip_model.get_image_features(**clip_inputs)
            clip_embeds = clip_embeds / clip_embeds.norm(dim=-1, keepdim=True)
            clip_sims = (clip_embeds @ self.clip_text_matrix.T).squeeze()
            clip_probs = torch.softmax(clip_sims * 100, dim=-1)
            
            # Get PLIP predictions
            plip_inputs = self.plip_processor(images=image, return_tensors="pt")
            plip_inputs = {k: v.to(self.device) for k, v in plip_inputs.items()}
            plip_embeds = self.plip_model.get_image_features(**plip_inputs)
            plip_embeds = plip_embeds / plip_embeds.norm(dim=-1, keepdim=True)
            plip_sims = (plip_embeds @ self.plip_text_matrix.T).squeeze()
            plip_probs = torch.softmax(plip_sims * 100, dim=-1)
            
            # Ensemble: use assigned model's probability for each class
            ensemble_probs = torch.zeros(len(self.classes), device=self.device)
            for i, cls in enumerate(self.classes):
                if self.model_assignment[cls] == 'clip':
                    ensemble_probs[i] = clip_probs[i]
                else:
                    ensemble_probs[i] = plip_probs[i]
            
            # Normalize
            ensemble_probs = ensemble_probs / ensemble_probs.sum()
            
            pred_idx = ensemble_probs.argmax().item()
            confidence = ensemble_probs[pred_idx].item()
            all_probs = {c: ensemble_probs[i].item() for i, c in enumerate(self.classes)}
        
        return self.classes[pred_idx], confidence, all_probs
    
    def classify_with_voting(self, image: Image.Image, mask: np.ndarray = None) -> tuple:
        """Classify using majority voting between CLIP and PLIP."""
        if mask is not None:
            image = self._crop_to_mask(image, mask)
            if image is None:
                return self.classes[0], 0.0, {}
        
        with torch.no_grad():
            # CLIP prediction
            clip_inputs = self.clip_processor(images=image, return_tensors="pt")
            clip_inputs = {k: v.to(self.device) for k, v in clip_inputs.items()}
            clip_embeds = self.clip_model.get_image_features(**clip_inputs)
            clip_embeds = clip_embeds / clip_embeds.norm(dim=-1, keepdim=True)
            clip_sims = (clip_embeds @ self.clip_text_matrix.T).squeeze()
            clip_probs = torch.softmax(clip_sims * 100, dim=-1)
            
            # PLIP prediction
            plip_inputs = self.plip_processor(images=image, return_tensors="pt")
            plip_inputs = {k: v.to(self.device) for k, v in plip_inputs.items()}
            plip_embeds = self.plip_model.get_image_features(**plip_inputs)
            plip_embeds = plip_embeds / plip_embeds.norm(dim=-1, keepdim=True)
            plip_sims = (plip_embeds @ self.plip_text_matrix.T).squeeze()
            plip_probs = torch.softmax(plip_sims * 100, dim=-1)
            
            # Average probabilities
            avg_probs = (clip_probs + plip_probs) / 2
            
            pred_idx = avg_probs.argmax().item()
            confidence = avg_probs[pred_idx].item()
            all_probs = {c: avg_probs[i].item() for i, c in enumerate(self.classes)}
        
        return self.classes[pred_idx], confidence, all_probs


def main():
    log("=" * 70)
    log("ENSEMBLE CLASSIFIER: CLIP + PLIP")
    log("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log(f"Device: {device}")
    
    # Load dataset
    log("\nLoading dataset...")
    image_dir = os.path.join(project_root, 'data', 'bcss', 'images')
    mask_dir = os.path.join(project_root, 'data', 'bcss', 'masks')
    dataset = BCSSDataset(image_dir=image_dir, mask_dir=mask_dir, split='test')
    log(f"Loaded {len(dataset)} samples")
    
    # Load ensemble classifier
    clip_path = os.environ.get('CLIP_MODEL_PATH', None)
    plip_path = os.environ.get('PLIP_MODEL_PATH', None)
    classifier = EnsembleClassifier(clip_path=clip_path, plip_path=plip_path, device=device)
    
    # Test both methods
    methods = {
        'class_assignment': classifier.classify,
        'probability_voting': classifier.classify_with_voting
    }
    
    class_map = {1: 'tumor', 2: 'stroma', 3: 'lymphocyte', 4: 'necrosis', 5: 'blood_vessel'}
    
    all_results = {}
    
    for method_name, classify_fn in methods.items():
        log(f"\n{'='*50}")
        log(f"Testing: {method_name}")
        log("=" * 50)
        
        results = {
            'correct': 0,
            'total': 0,
            'per_class': {c: {'correct': 0, 'total': 0} for c in classifier.classes}
        }
        
        pbar = tqdm(range(len(dataset)), desc=method_name, file=sys.stdout, ncols=80)
        
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
                
                pred_class, conf, probs = classify_fn(pil_image, class_mask)
                
                results['total'] += 1
                results['per_class'][class_name]['total'] += 1
                
                if pred_class == class_name:
                    results['correct'] += 1
                    results['per_class'][class_name]['correct'] += 1
            
            if results['total'] > 0:
                acc = results['correct'] / results['total'] * 100
                pbar.set_postfix({'acc': f'{acc:.1f}%'})
        
        overall_acc = results['correct'] / results['total'] if results['total'] > 0 else 0
        
        log(f"\n{method_name} Results:")
        log(f"  Overall: {overall_acc:.2%} ({results['correct']}/{results['total']})")
        
        for cls in classifier.classes:
            c = results['per_class'][cls]
            acc = c['correct'] / c['total'] if c['total'] > 0 else 0
            log(f"  {cls:15s}: {acc:6.2%} ({c['correct']}/{c['total']})")
        
        all_results[method_name] = {
            'accuracy': overall_acc,
            'per_class': {
                cls: {
                    'accuracy': results['per_class'][cls]['correct'] / results['per_class'][cls]['total']
                        if results['per_class'][cls]['total'] > 0 else 0,
                    'correct': results['per_class'][cls]['correct'],
                    'total': results['per_class'][cls]['total']
                }
                for cls in classifier.classes
            }
        }
    
    # Summary
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)
    log(f"\nBaselines:")
    log(f"  CLIP only:     38.9%")
    log(f"  PLIP only:     26.9%")
    log(f"\nEnsemble:")
    for method, res in all_results.items():
        log(f"  {method:20s}: {res['accuracy']:.1%}")
    
    # Save results
    output_dir = os.path.join(project_root, 'results', 'ensemble_clip_plip')
    os.makedirs(output_dir, exist_ok=True)
    
    output = {
        'methods': all_results,
        'baselines': {'clip': 0.389, 'plip': 0.269},
        'timestamp': datetime.now().isoformat()
    }
    
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(output, f, indent=2)
    
    log(f"\nResults saved to: {output_dir}/")


if __name__ == '__main__':
    main()
