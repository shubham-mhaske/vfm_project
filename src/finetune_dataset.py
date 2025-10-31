import os
import numpy as np
import torch
from PIL import Image
from sam2.training.dataset.vos_raw_dataset import VOSRawDataset, VOSFrame, VOSVideo

class BCSSSegmentLoader:
    def __init__(self, mask_path):
        """SegmentLoader for a single BCSS mask file."""
        self.mask_path = mask_path

    def load(self, frame_id):
        """
        Loads the single mask file and converts it into binary masks for each class.
        frame_id is unused since we have single images, but required by the API.
        """
        masks = np.array(Image.open(self.mask_path))
        
        object_ids = np.unique(masks)
        object_ids = object_ids[object_ids != 0]  # remove background (0)

        binary_segments = {}
        for i in object_ids:
            bs = (masks == i)
            binary_segments[i] = torch.from_numpy(bs)

        return binary_segments

class BCSSRawDataset(VOSRawDataset):
    def __init__(self, img_folder, gt_folder, split='train'):
        self.img_folder = img_folder
        self.gt_folder = gt_folder
        
        all_files = sorted([f for f in os.listdir(img_folder) if f.endswith('.png')])
        
        test_prefixes = ['TCGA-OL-', 'TCGA-LL-', 'TCGA-E2-', 'TCGA-EW-', 'TCGA-GM-', 'TCGA-S3-']
        test_files = [f for f in all_files if any(f.startswith(p) for p in test_prefixes)]
        train_val_files = [f for f in all_files if f not in test_files]
        
        np.random.seed(42)
        np.random.shuffle(train_val_files)
        val_split = int(0.2 * len(train_val_files))
        val_files = train_val_files[:val_split]
        train_files = train_val_files[val_split:]
        
        if split == 'train':
            self.image_names = train_files
        elif split == 'val':
            self.image_names = val_files
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train' or 'val'.")

    def get_video(self, idx):
        image_name = self.image_names[idx]
        
        image_path = os.path.join(self.img_folder, image_name)
        mask_path = os.path.join(self.gt_folder, image_name)

        frames = [VOSFrame(frame_idx=0, image_path=image_path)]
        video = VOSVideo(video_name=os.path.splitext(image_name)[0], video_id=idx, frames=frames)
        
        segment_loader = BCSSSegmentLoader(mask_path)
        
        return video, segment_loader

    def __len__(self):
        return len(self.image_names)