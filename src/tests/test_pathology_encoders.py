#!/usr/bin/env python3
"""
Test UNI encoder for pathology image classification.
UNI is a pathology foundation model trained on 100M+ histopathology images.

Expected: Significant improvement over CLIP (potentially 50-70%)
"""

import os
import sys
import json
import torch
import numpy as np
from PIL import Image
from datetime import datetime
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'sam2'))

from dataset import BCSSDataset


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
    sys.stdout.flush()


class UNIFeatureExtractor:
    """Extract features using UNI pathology foundation model."""
    
    def __init__(self, device='cuda'):
        self.device = device
        
        try:
            import timm
            from timm.data import resolve_data_config
            from timm.data.transforms_factory import create_transform
            
            log("Loading UNI model...")
            
            # UNI model path
            uni_path = os.environ.get('UNI_MODEL_PATH', 
                os.path.join(project_root, 'models', 'uni_model'))
            
            if os.path.exists(uni_path):
                log(f"  Loading from: {uni_path}")
                self.model = timm.create_model(
                    'vit_large_patch16_224',
                    pretrained=False,
                    num_classes=0  # Remove classification head
                )
                # Load weights
                weight_path = os.path.join(uni_path, 'pytorch_model.bin')
                if os.path.exists(weight_path):
                    state_dict = torch.load(weight_path, map_location=device)
                    self.model.load_state_dict(state_dict, strict=False)
            else:
                log("  UNI weights not found, using ImageNet pretrained ViT-L")
                self.model = timm.create_model(
                    'vit_large_patch16_224',
                    pretrained=True,
                    num_classes=0
                )
            
            self.model.to(device)
            self.model.eval()
            
            # Get transforms
            config = resolve_data_config({}, model=self.model)
            self.transform = create_transform(**config)
            
            log("  UNI loaded!")
            
        except Exception as e:
            log(f"Error loading UNI: {e}")
            log("Falling back to CLIP features...")
            raise
    
    def extract_features(self, image: Image.Image, mask: np.ndarray = None) -> np.ndarray:
        """Extract UNI features for an image region."""
        if mask is not None:
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
            
            image = image.crop((cmin, rmin, cmax, rmax))
        
        with torch.no_grad():
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            features = self.model(img_tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        
        return features.cpu().numpy().squeeze()


class CTransPathExtractor:
    """Extract features using CTransPath pathology encoder."""
    
    def __init__(self, device='cuda'):
        self.device = device
        
        try:
            import timm
            from timm.data import resolve_data_config
            from timm.data.transforms_factory import create_transform
            
            log("Loading CTransPath model...")
            
            ctranspath_path = os.environ.get('CTRANSPATH_MODEL_PATH',
                os.path.join(project_root, 'models', 'ctranspath'))
            
            weight_file = os.path.join(ctranspath_path, 'ctranspath.pth')
            
            if os.path.exists(weight_file):
                log(f"  Loading from: {weight_file}")
                self.model = timm.create_model(
                    'swin_tiny_patch4_window7_224',
                    pretrained=False,
                    num_classes=0
                )
                state_dict = torch.load(weight_file, map_location=device)
                if 'model' in state_dict:
                    state_dict = state_dict['model']
                self.model.load_state_dict(state_dict, strict=False)
            else:
                log("  CTransPath weights not found, using ImageNet pretrained Swin-T")
                self.model = timm.create_model(
                    'swin_tiny_patch4_window7_224',
                    pretrained=True,
                    num_classes=0
                )
            
            self.model.to(device)
            self.model.eval()
            
            config = resolve_data_config({}, model=self.model)
            self.transform = create_transform(**config)
            
            log("  CTransPath loaded!")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load CTransPath: {e}")
    
    def extract_features(self, image: Image.Image, mask: np.ndarray = None) -> np.ndarray:
        """Extract features for an image region."""
        if mask is not None:
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
            
            image = image.crop((cmin, rmin, cmax, rmax))
        
        with torch.no_grad():
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            features = self.model(img_tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        
        return features.cpu().numpy().squeeze()


def extract_dataset_features(extractor, dataset, split_name):
    """Extract features for all samples."""
    log(f"\nExtracting features for {split_name} set ({len(dataset)} images)...")
    
    features = []
    labels = []
    
    class_map = {1: 'tumor', 2: 'stroma', 3: 'lymphocyte', 4: 'necrosis', 5: 'blood_vessel'}
    class_to_idx = {name: idx for idx, name in enumerate(class_map.values())}
    
    pbar = tqdm(range(len(dataset)), desc=f"Extracting {split_name}", 
                file=sys.stdout, ncols=80)
    
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
            if class_name is None:
                continue
            
            class_mask = (mask == class_idx)
            if not class_mask.any():
                continue
            
            feat = extractor.extract_features(pil_image, class_mask)
            if feat is not None:
                features.append(feat)
                labels.append(class_to_idx[class_name])
    
    return np.array(features), np.array(labels)


def train_and_evaluate(X_train, y_train, X_test, y_test, class_names, model_name):
    """Train LogisticRegression and evaluate."""
    log(f"\n{'='*50}")
    log(f"Training classifier on {model_name} features")
    log("=" * 50)
    
    # Normalize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train
    log("Training Logistic Regression...")
    clf = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced', 
                             random_state=42, n_jobs=-1)
    clf.fit(X_train_scaled, y_train)
    
    # Evaluate
    preds = clf.predict(X_test_scaled)
    acc = (preds == y_test).mean()
    
    log(f"Accuracy: {acc:.2%}")
    
    # Per-class
    log("\nPer-Class Breakdown:")
    per_class = {}
    for i, cls in enumerate(class_names):
        mask = (y_test == i)
        if mask.sum() > 0:
            cls_acc = (preds[mask] == y_test[mask]).mean()
            log(f"  {cls:15s}: {cls_acc:6.2%} ({mask.sum()} samples)")
            per_class[cls] = {'accuracy': float(cls_acc), 'total': int(mask.sum())}
    
    return {
        'accuracy': float(acc),
        'per_class': per_class
    }


def main():
    log("=" * 70)
    log("PATHOLOGY ENCODER COMPARISON")
    log("UNI vs CTransPath vs CLIP features")
    log("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log(f"Device: {device}")
    
    # Paths
    image_dir = os.path.join(project_root, 'data', 'bcss', 'images')
    mask_dir = os.path.join(project_root, 'data', 'bcss', 'masks')
    
    # Load datasets
    log("\nLoading datasets...")
    train_dataset = BCSSDataset(image_dir=image_dir, mask_dir=mask_dir, split='train')
    test_dataset = BCSSDataset(image_dir=image_dir, mask_dir=mask_dir, split='test')
    log(f"  Train: {len(train_dataset)} images")
    log(f"  Test: {len(test_dataset)} images")
    
    class_names = ['tumor', 'stroma', 'lymphocyte', 'necrosis', 'blood_vessel']
    
    results = {}
    
    # Try CTransPath (more likely to be available)
    try:
        log("\n" + "-" * 50)
        log("Testing CTransPath encoder...")
        ctranspath = CTransPathExtractor(device=device)
        
        X_train, y_train = extract_dataset_features(ctranspath, train_dataset, "train")
        X_test, y_test = extract_dataset_features(ctranspath, test_dataset, "test")
        
        results['ctranspath'] = train_and_evaluate(
            X_train, y_train, X_test, y_test, class_names, "CTransPath"
        )
    except Exception as e:
        log(f"CTransPath failed: {e}")
    
    # Try UNI
    try:
        log("\n" + "-" * 50)
        log("Testing UNI encoder...")
        uni = UNIFeatureExtractor(device=device)
        
        X_train, y_train = extract_dataset_features(uni, train_dataset, "train")
        X_test, y_test = extract_dataset_features(uni, test_dataset, "test")
        
        results['uni'] = train_and_evaluate(
            X_train, y_train, X_test, y_test, class_names, "UNI"
        )
    except Exception as e:
        log(f"UNI failed: {e}")
    
    # Summary
    log("\n" + "=" * 70)
    log("FINAL COMPARISON")
    log("=" * 70)
    
    log("\nBaselines:")
    log("  CLIP zero-shot:     38.9%")
    log("  CLIP + LR:          40.4%")
    
    log("\nPathology Encoders:")
    for name, res in results.items():
        log(f"  {name:15s} + LR: {res['accuracy']:.1%}")
    
    # Save results
    output_dir = os.path.join(project_root, 'results', 'pathology_encoders')
    os.makedirs(output_dir, exist_ok=True)
    
    output = {
        'results': results,
        'baselines': {
            'clip_zeroshot': 0.389,
            'clip_lr': 0.404
        },
        'timestamp': datetime.now().isoformat()
    }
    
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(output, f, indent=2)
    
    log(f"\nResults saved to: {output_dir}/")


if __name__ == '__main__':
    main()
