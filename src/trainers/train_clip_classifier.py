#!/usr/bin/env python3
"""
Train a simple classifier on CLIP features.

Instead of zero-shot classification, we:
1. Extract CLIP image embeddings for all training samples
2. Train a LogisticRegression/MLP classifier
3. Evaluate on test set

Expected: 50-65% accuracy (vs 38.9% zero-shot)
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
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import pickle

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'sam2'))

from dataset import BCSSDataset


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
    sys.stdout.flush()


class CLIPFeatureExtractor:
    """Extract CLIP image features using transformers library."""
    
    def __init__(self, device='cuda', model_path=None):
        self.device = device
        
        from transformers import CLIPProcessor, CLIPModel
        
        if model_path and os.path.exists(model_path):
            log(f"  Loading CLIP from: {model_path}")
            self.model = CLIPModel.from_pretrained(model_path, local_files_only=True)
            self.processor = CLIPProcessor.from_pretrained(model_path, local_files_only=True)
        else:
            log("  Loading CLIP from HuggingFace...")
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        self.model.to(device)
        self.model.eval()
        log("  CLIP loaded!")
    
    def extract_features(self, image: Image.Image, mask: np.ndarray = None) -> np.ndarray:
        """
        Extract CLIP features for an image region.
        
        Args:
            image: PIL Image
            mask: Optional binary mask to crop region
            
        Returns:
            Feature vector (512-dim for ViT-B/32)
        """
        # Crop to mask if provided
        if mask is not None:
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            if not rows.any() or not cols.any():
                return None
            
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            
            # Add padding
            h, w = mask.shape
            pad = 10
            rmin, rmax = max(0, rmin - pad), min(h, rmax + pad)
            cmin, cmax = max(0, cmin - pad), min(w, cmax + pad)
            
            image = image.crop((cmin, rmin, cmax, rmax))
        
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            features = self.model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
            
        return features.cpu().numpy().squeeze()


def extract_dataset_features(extractor, dataset, split_name):
    """Extract features for all samples in a dataset."""
    log(f"\nExtracting features for {split_name} set ({len(dataset)} images)...")
    
    features = []
    labels = []
    
    class_map = {1: 'tumor', 2: 'stroma', 3: 'lymphocyte', 4: 'necrosis', 5: 'blood_vessel'}
    class_to_idx = {name: idx for idx, name in enumerate(class_map.values())}
    
    pbar = tqdm(range(len(dataset)), desc=f"Extracting {split_name}", 
                file=sys.stdout, ncols=80)
    
    for idx in pbar:
        sample = dataset[idx]
        image_np = sample.get('image_np', None)
        mask = sample['mask']
        
        if image_np is None:
            image = sample['image']
            if isinstance(image, torch.Tensor):
                image = image.numpy()
                if image.shape[0] == 3:
                    image = image.transpose(1, 2, 0)
            image_np = image
        
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
        
        if (idx + 1) % 10 == 0:
            sys.stdout.flush()
    
    return np.array(features), np.array(labels)


def main():
    log("=" * 70)
    log("CLIP FEATURE CLASSIFIER")
    log("Train classifier on CLIP embeddings")
    log("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log(f"Device: {device}")
    if device == 'cuda':
        log(f"GPU: {torch.cuda.get_device_name(0)}")
    sys.stdout.flush()
    
    # Paths
    image_dir = os.path.join(project_root, 'data', 'bcss', 'images')
    mask_dir = os.path.join(project_root, 'data', 'bcss', 'masks')
    
    # Load datasets
    log("\nLoading datasets...")
    train_dataset = BCSSDataset(image_dir=image_dir, mask_dir=mask_dir, split='train')
    test_dataset = BCSSDataset(image_dir=image_dir, mask_dir=mask_dir, split='test')
    log(f"  Train: {len(train_dataset)} images")
    log(f"  Test: {len(test_dataset)} images")
    sys.stdout.flush()
    
    # Load CLIP
    log("\nLoading CLIP feature extractor...")
    clip_path = os.environ.get('CLIP_MODEL_PATH', None)
    extractor = CLIPFeatureExtractor(device=device, model_path=clip_path)
    
    # Extract features
    X_train, y_train = extract_dataset_features(extractor, train_dataset, "train")
    X_test, y_test = extract_dataset_features(extractor, test_dataset, "test")
    
    log(f"\nFeature shapes:")
    log(f"  Train: {X_train.shape} features, {len(y_train)} labels")
    log(f"  Test: {X_test.shape} features, {len(y_test)} labels")
    sys.stdout.flush()
    
    # Class names for reporting
    class_names = ['tumor', 'stroma', 'lymphocyte', 'necrosis', 'blood_vessel']
    
    # Normalize features
    log("\nNormalizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    
    # Train Logistic Regression
    log("\n" + "-" * 50)
    log("Training Logistic Regression...")
    sys.stdout.flush()
    
    lr = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced', random_state=42, n_jobs=-1)
    lr.fit(X_train_scaled, y_train)
    
    lr_preds = lr.predict(X_test_scaled)
    lr_acc = (lr_preds == y_test).mean()
    log(f"Logistic Regression Accuracy: {lr_acc:.2%}")
    
    results['logistic_regression'] = {
        'accuracy': float(lr_acc),
        'report': classification_report(y_test, lr_preds, target_names=class_names, output_dict=True)
    }
    
    # Train MLP
    log("\n" + "-" * 50)
    log("Training MLP Classifier...")
    sys.stdout.flush()
    
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42
    )
    mlp.fit(X_train_scaled, y_train)
    
    mlp_preds = mlp.predict(X_test_scaled)
    mlp_acc = (mlp_preds == y_test).mean()
    log(f"MLP Accuracy: {mlp_acc:.2%}")
    
    results['mlp'] = {
        'accuracy': float(mlp_acc),
        'report': classification_report(y_test, mlp_preds, target_names=class_names, output_dict=True)
    }
    
    # Summary
    log("\n" + "=" * 70)
    log("RESULTS SUMMARY")
    log("=" * 70)
    
    log(f"\nZero-shot CLIP:        38.9%")
    log(f"Logistic Regression:   {lr_acc:.1%}")
    log(f"MLP Classifier:        {mlp_acc:.1%}")
    
    best_method = 'logistic_regression' if lr_acc >= mlp_acc else 'mlp'
    best_acc = max(lr_acc, mlp_acc)
    improvement = best_acc - 0.389
    
    log(f"\nBest: {best_method} ({best_acc:.1%})")
    log(f"Improvement over zero-shot: {improvement:+.1%}")
    
    # Per-class breakdown for best method
    best_preds = lr_preds if best_method == 'logistic_regression' else mlp_preds
    log("\nPer-Class Breakdown (Best Method):")
    log("-" * 50)
    for i, class_name in enumerate(class_names):
        class_mask = (y_test == i)
        if class_mask.sum() > 0:
            class_acc = (best_preds[class_mask] == y_test[class_mask]).mean()
            log(f"  {class_name:15s}: {class_acc:6.2%} ({class_mask.sum()} samples)")
    
    # Save results
    output_dir = os.path.join(project_root, 'results', 'clip_classifier')
    os.makedirs(output_dir, exist_ok=True)
    
    output = {
        'method': 'CLIP Feature Classifier',
        'results': results,
        'best_method': best_method,
        'best_accuracy': float(best_acc),
        'comparison': {
            'zero_shot_clip': 0.389,
            'improvement': float(improvement)
        },
        'train_samples': len(y_train),
        'test_samples': len(y_test),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(output, f, indent=2)
    
    # Save best model
    best_model = lr if best_method == 'logistic_regression' else mlp
    with open(os.path.join(output_dir, 'best_model.pkl'), 'wb') as f:
        pickle.dump({'model': best_model, 'scaler': scaler}, f)
    
    log(f"\nResults saved to: {output_dir}/")
    log("Done!")


if __name__ == '__main__':
    main()
