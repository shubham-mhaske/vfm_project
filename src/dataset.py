import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2
import os

class BCSSDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, split='train'):
        """
        Args:
            image_dir: Directory containing RGB images
            mask_dir: Directory containing segmentation masks
            transform: Optional image transformations
            split: 'train', 'val', or 'test'
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        all_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        
        # Define test set prefixes
        test_prefixes = ['TCGA-OL-', 'TCGA-LL-', 'TCGA-E2-', 'TCGA-EW-', 'TCGA-GM-', 'TCGA-S3-']
        
        test_files = [f for f in all_files if any(f.startswith(p) for p in test_prefixes)]
        train_val_files = [f for f in all_files if f not in test_files]
        
        # Split train_val_files into training and validation sets
        # We use a fixed random seed for reproducibility
        np.random.seed(42)
        np.random.shuffle(train_val_files)
        val_split = int(0.2 * len(train_val_files))
        val_files = train_val_files[:val_split]
        train_files = train_val_files[val_split:]
        
        if split == 'train':
            self.image_files = train_files
        elif split == 'val':
            self.image_files = val_files
        elif split == 'test':
            self.image_files = test_files
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'.")

        # Define tissue class mapping
        # Note: BCSS uses class ID 18 for blood_vessel, not 5
        # Class 5 in BCSS is actually 'glandular_secretions'
        self.class_names = {
            0: 'background',
            1: 'tumor',
            2: 'stroma',
            3: 'lymphocyte',
            4: 'necrosis',
            18: 'blood_vessel'  # Fixed: was incorrectly mapped to 5
        }
        
        # Target classes for evaluation (excluding background)
        self.target_class_ids = {1, 2, 3, 4, 18}
    
    def __len__(self):
        return len(self.image_files)
    
    def get_centroid_prompt(self, mask, class_id):
        """Calculate centroid of a specific class in the mask"""
        class_mask = (mask == class_id).astype(np.uint8)
        
        if class_mask.sum() == 0:
            return None
        
        # Calculate centroid
        moments = cv2.moments(class_mask)
        if moments['m00'] == 0:
            return None
            
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        
        return (cx, cy), class_mask
    
    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        # Load mask
        mask_path = os.path.join(self.mask_dir, self.image_files[idx])
        mask = np.array(Image.open(mask_path))
        
        # Get all unique classes in this mask (excluding background)
        unique_classes = np.unique(mask)
        unique_classes = unique_classes[unique_classes != 0]
        
        image_np = np.array(image) # Keep a numpy copy for things like centroid calculation if needed before transform

        if self.transform:
            image = self.transform(image)
        else:
            # Default transform to tensor
            image = torch.from_numpy(image_np.transpose((2, 0, 1))).float()

        mask = torch.from_numpy(mask).long()
        
        return {
            'image': image,
            'mask': mask,
            'unique_classes': unique_classes,
            'filename': self.image_files[idx],
            'image_np': image_np
        }