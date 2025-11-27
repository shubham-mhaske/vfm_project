import os
import numpy as np
import torch
from PIL import Image
from training.dataset.vos_raw_dataset import VOSRawDataset, VOSFrame, VOSVideo

# Use the project's prompt helper to generate prompts from masks
from src.sam_segmentation import get_prompts_from_mask

# Target classes for training - matches evaluation classes
# BCSS class IDs: 1=tumor, 2=stroma, 3=lymphocyte, 4=necrosis, 18=blood_vessel
TARGET_CLASS_IDS = {1, 2, 3, 4, 18}

class BCSSSegmentLoader:
    def __init__(self, mask_path, prompt_type: str = "centroid", use_neg_points: bool = False, num_points: int = 5, filter_classes: bool = True):
        """SegmentLoader for a single BCSS mask file that also prepares prompts per object.

        Args:
            mask_path: Path to the palettized mask image.
            prompt_type: One of {"centroid", "box", "multi_point"}.
            use_neg_points: Whether to include negative clicks sampled outside the bbox (if available).
            num_points: Number of positive points to sample for multi_point prompts.
            filter_classes: If True, only load target classes (1,2,3,4,18). Default True.
        """
        self.mask_path = mask_path
        self.prompt_type = prompt_type
        self.use_neg_points = use_neg_points
        self.num_points = num_points
        self.filter_classes = filter_classes
        self._last_prompts = None  # Mapping: obj_id -> {point_coords, point_labels}

    def load(self, frame_id):
        """
        Loads the single mask file and converts it into binary masks for each class.
        frame_id is unused since we have single images, but required by the API.
        """
        masks = np.array(Image.open(self.mask_path))
        
        object_ids = np.unique(masks)
        object_ids = object_ids[object_ids != 0]  # remove background (0)
        
        # Filter to only target classes if enabled
        if self.filter_classes:
            object_ids = np.array([oid for oid in object_ids if oid in TARGET_CLASS_IDS])

        binary_segments = {}
        prompts_per_obj = {}
        # Generate per-object binary mask and corresponding prompts
        for i in object_ids:
            bs = (masks == i)
            binary_segments[i] = torch.from_numpy(bs)

            # Build prompts for this object mask using helper
            prompt_dict = get_prompts_from_mask(bs.astype(np.uint8), num_points=self.num_points)

            # Normalize to point-based inputs expected by SAM2 training: two tensors (coords, labels)
            point_coords = None
            point_labels = None
            
            # Handle mixed prompt type
            current_prompt_type = self.prompt_type
            if self.prompt_type == 'mixed':
                # Randomly choose between box and centroid
                current_prompt_type = 'box' if np.random.rand() > 0.5 else 'centroid'

            if current_prompt_type == 'centroid' and 'centroid' in prompt_dict:
                point_coords, point_labels = prompt_dict['centroid']
            elif current_prompt_type == 'multi_point' and 'multi_point' in prompt_dict:
                point_coords, point_labels = prompt_dict['multi_point']
            elif current_prompt_type == 'box' and 'box' in prompt_dict:
                # SAM2 uses two special labels for box corners: 2 (top-left) and 3 (bottom-right)
                box = prompt_dict['box']  # shape (2,2) [[x0,y0],[x1,y1]]
                point_coords = box
                point_labels = np.array([2, 3], dtype=np.int32)

            # Optionally append negative points if requested and available
            if self.use_neg_points and 'neg_points' in prompt_dict:
                neg_coords, neg_labels = prompt_dict['neg_points']
                if point_coords is not None:
                    point_coords = np.concatenate([point_coords, neg_coords], axis=0)
                    point_labels = np.concatenate([point_labels, neg_labels], axis=0)
                else:
                    point_coords, point_labels = neg_coords, neg_labels

            if point_coords is not None and point_labels is not None:
                prompts_per_obj[i] = {
                    'point_coords': torch.as_tensor(point_coords, dtype=torch.float32),
                    'point_labels': torch.as_tensor(point_labels, dtype=torch.int32),
                }
            else:
                # If prompt generation failed for this object, record an empty prompt
                prompts_per_obj[i] = {
                    'point_coords': None,
                    'point_labels': None,
                }

        # Store so the VOSDataset can retrieve prompts aligned with this load
        self._last_prompts = prompts_per_obj

        return binary_segments

    def get_prompts_for_last_load(self):
        """Returns a mapping obj_id -> {'point_coords': Tensor|None, 'point_labels': Tensor|None}
        aligned with the most recent call to load()."""
        return self._last_prompts or {}

class BCSSRawDataset(VOSRawDataset):
    def __init__(self, img_folder, gt_folder, split='train', prompt_type: str = 'centroid', use_neg_points: bool = False, num_points: int = 5, filter_classes: bool = True):
        self.img_folder = img_folder
        self.gt_folder = gt_folder
        self.prompt_type = prompt_type
        self.use_neg_points = use_neg_points
        self.num_points = num_points
        self.filter_classes = filter_classes  # Filter to target classes only
        
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
        
        segment_loader = BCSSSegmentLoader(
            mask_path,
            prompt_type=self.prompt_type,
            use_neg_points=self.use_neg_points,
            num_points=self.num_points,
            filter_classes=self.filter_classes,
        )
        
        return video, segment_loader

    def __len__(self):
        return len(self.image_names)