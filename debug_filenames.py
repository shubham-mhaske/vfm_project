
import os

image_dir = 'CrowdsourcingDataset-Amgadetal2019/data/images'
mask_dir = 'CrowdsourcingDataset-Amgadetal2019/data/masks'

image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png') and not f.startswith('.')])

print("Image files:")
print(image_files)
print("\nMask files:")
print(mask_files)

if image_files == mask_files:
    print("\nFile lists are identical.")
else:
    print("\nFile lists are NOT identical.")

    # Find the differences
    image_set = set(image_files)
    mask_set = set(mask_files)

    print("\nFiles in images but not in masks:")
    print(image_set - mask_set)

    print("\nFiles in masks but not in images:")
    print(mask_set - image_set)

