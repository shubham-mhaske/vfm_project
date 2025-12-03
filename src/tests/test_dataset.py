
import torch
import numpy as np
from dataset import BCSSDataset

def test_dataset_loading():
    """Tests the BCSSDataset loader."""
    image_dir = 'data/bcss/images'
    mask_dir = 'data/bcss/masks'
    
    # Initialize the dataset
    bcss_dataset = BCSSDataset(image_dir=image_dir, mask_dir=mask_dir)

    # Check the length of the dataset
    print(f"Dataset length: {len(bcss_dataset)}")

    # Get a sample from the dataset
    sample = bcss_dataset[0]

    # Check the types and shapes of the loaded data
    image, mask, filename = sample['image'], sample['mask'], sample['filename']
    print(f"Sample filename: {filename}")
    print(f"Image type: {type(image)}")
    print(f"Mask type: {type(mask)}")
    print(f"Mask shape: {mask.shape}")
    print(f"Unique values in mask: {np.unique(mask)}")

if __name__ == '__main__':
    test_dataset_loading()
