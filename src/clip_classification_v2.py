"""
Improved CLIP Classification Module v2
=======================================
Key improvements over v1:
1. Multiple cropping strategies (tight crop, context crop, multi-scale)
2. Mask overlay options (transparent, blur, none instead of white)
3. Confidence thresholding with "uncertain" class
4. Ensemble of crops for robust prediction
5. Per-class calibration using prompts

Author: VFM Project
"""
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image, ImageFilter
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional
import torch.nn.functional as F


# Default local path for offline CLIP model on HPRC
DEFAULT_CLIP_LOCAL_PATH = os.environ.get('CLIP_MODEL_PATH', None)


def get_clip_model_path():
    """Get CLIP model path - local if available, otherwise HuggingFace."""
    if DEFAULT_CLIP_LOCAL_PATH and os.path.exists(DEFAULT_CLIP_LOCAL_PATH):
        return DEFAULT_CLIP_LOCAL_PATH
    
    scratch = os.environ.get('SCRATCH', '')
    local_paths = [
        os.path.join(scratch, 'clip_model'),
        os.path.join(scratch, 'clip_cache'),
        os.path.expanduser('~/clip_model'),
    ]
    
    for path in local_paths:
        if os.path.exists(path) and os.path.exists(os.path.join(path, 'config.json')):
            print(f"Using local CLIP model from: {path}")
            return path
    
    return "openai/clip-vit-base-patch32"


def load_prompts_from_json(json_path: str) -> Dict[str, List[str]]:
    """Loads a prompt dictionary from a JSON file."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Prompt file not found: {json_path}")
    with open(json_path, 'r') as f:
        return json.load(f)


class CropStrategy:
    """Different strategies for cropping regions from masks."""
    
    @staticmethod
    def tight_crop(image_np: np.ndarray, mask: np.ndarray, 
                   padding: int = 0) -> Tuple[Optional[Image.Image], np.ndarray]:
        """
        Tight bounding box crop around mask.
        Returns: (cropped_image, cropped_mask)
        """
        if mask.sum() == 0:
            return None, mask
        
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not np.any(rows) or not np.any(cols):
            return None, mask
        
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        # Add padding with bounds checking
        h, w = mask.shape
        rmin = max(0, rmin - padding)
        rmax = min(h - 1, rmax + padding)
        cmin = max(0, cmin - padding)
        cmax = min(w - 1, cmax + padding)
        
        cropped_image = image_np[rmin:rmax+1, cmin:cmax+1, :]
        cropped_mask = mask[rmin:rmax+1, cmin:cmax+1]
        
        return Image.fromarray(cropped_image), cropped_mask
    
    @staticmethod
    def context_crop(image_np: np.ndarray, mask: np.ndarray,
                     context_ratio: float = 0.3) -> Tuple[Optional[Image.Image], np.ndarray]:
        """
        Crop with additional context around the mask.
        context_ratio: fraction of bounding box size to add as context
        """
        if mask.sum() == 0:
            return None, mask
        
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not np.any(rows) or not np.any(cols):
            return None, mask
        
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        # Calculate context padding
        height = rmax - rmin
        width = cmax - cmin
        pad_h = int(height * context_ratio)
        pad_w = int(width * context_ratio)
        
        h, w = mask.shape
        rmin = max(0, rmin - pad_h)
        rmax = min(h - 1, rmax + pad_h)
        cmin = max(0, cmin - pad_w)
        cmax = min(w - 1, cmax + pad_w)
        
        cropped_image = image_np[rmin:rmax+1, cmin:cmax+1, :]
        cropped_mask = mask[rmin:rmax+1, cmin:cmax+1]
        
        return Image.fromarray(cropped_image), cropped_mask
    
    @staticmethod
    def square_crop(image_np: np.ndarray, mask: np.ndarray,
                    target_size: int = 224) -> Tuple[Optional[Image.Image], np.ndarray]:
        """
        Create square crop centered on mask centroid.
        Useful for ensuring consistent aspect ratio for CLIP.
        """
        if mask.sum() == 0:
            return None, mask
        
        # Find centroid
        y_coords, x_coords = np.where(mask)
        cy, cx = int(np.mean(y_coords)), int(np.mean(x_coords))
        
        h, w = mask.shape
        half_size = target_size // 2
        
        # Calculate bounds with clamping
        rmin = max(0, cy - half_size)
        rmax = min(h, cy + half_size)
        cmin = max(0, cx - half_size)
        cmax = min(w, cx + half_size)
        
        # Adjust if we hit boundaries
        if rmax - rmin < target_size and rmin == 0:
            rmax = min(h, target_size)
        elif rmax - rmin < target_size:
            rmin = max(0, rmax - target_size)
        
        if cmax - cmin < target_size and cmin == 0:
            cmax = min(w, target_size)
        elif cmax - cmin < target_size:
            cmin = max(0, cmax - target_size)
        
        cropped_image = image_np[rmin:rmax, cmin:cmax, :]
        cropped_mask = mask[rmin:rmax, cmin:cmax]
        
        return Image.fromarray(cropped_image), cropped_mask


class MaskOverlay:
    """Different strategies for handling non-mask regions."""
    
    @staticmethod
    def no_overlay(image: Image.Image, mask: np.ndarray) -> Image.Image:
        """Return image as-is (show full context)."""
        return image
    
    @staticmethod
    def blur_background(image: Image.Image, mask: np.ndarray, 
                        blur_radius: int = 15) -> Image.Image:
        """Blur regions outside the mask to de-emphasize background."""
        img_np = np.array(image)
        
        # Create blurred version
        blurred = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        blurred_np = np.array(blurred)
        
        # Composite: sharp inside mask, blurred outside
        mask_3d = np.stack([mask] * 3, axis=-1).astype(bool)
        result = np.where(mask_3d, img_np, blurred_np)
        
        return Image.fromarray(result.astype(np.uint8))
    
    @staticmethod
    def dim_background(image: Image.Image, mask: np.ndarray,
                       dim_factor: float = 0.3) -> Image.Image:
        """Dim regions outside the mask."""
        img_np = np.array(image).astype(float)
        
        mask_3d = np.stack([mask] * 3, axis=-1).astype(bool)
        result = np.where(mask_3d, img_np, img_np * dim_factor)
        
        return Image.fromarray(result.astype(np.uint8))
    
    @staticmethod
    def highlight_boundary(image: Image.Image, mask: np.ndarray,
                           color: Tuple[int, int, int] = (255, 255, 0),
                           thickness: int = 2) -> Image.Image:
        """Add colored boundary around mask region."""
        import cv2
        img_np = np.array(image)
        
        # Find contours
        contours, _ = cv2.findContours(mask.astype(np.uint8), 
                                        cv2.RETR_EXTERNAL, 
                                        cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours
        cv2.drawContours(img_np, contours, -1, color, thickness)
        
        return Image.fromarray(img_np)
    
    @staticmethod
    def masked_only(image: Image.Image, mask: np.ndarray,
                    bg_color: Tuple[int, int, int] = (128, 128, 128)) -> Image.Image:
        """Show only masked region with neutral gray background."""
        img_np = np.array(image)
        
        mask_3d = np.stack([mask] * 3, axis=-1).astype(bool)
        bg = np.full_like(img_np, bg_color)
        result = np.where(mask_3d, img_np, bg)
        
        return Image.fromarray(result.astype(np.uint8))


class CLIPClassifierV2:
    """
    Improved CLIP Classifier with multiple enhancements:
    - Multi-crop ensemble
    - Multiple overlay strategies
    - Confidence thresholding
    - Temperature-scaled predictions
    """
    
    def __init__(self, model_name: str = None, device: str = None,
                 temperature: float = 1.0,
                 confidence_threshold: float = 0.0):
        """
        Args:
            model_name: CLIP model name or local path
            device: torch device
            temperature: softmax temperature for calibration (>1 = smoother)
            confidence_threshold: minimum confidence to accept prediction
        """
        self.device = device if device else "cpu"
        self.temperature = temperature
        self.confidence_threshold = confidence_threshold
        
        if model_name is None:
            model_name = get_clip_model_path()
        
        print(f"Loading CLIP model from: {model_name}")
        is_local = os.path.exists(model_name)
        self.model = CLIPModel.from_pretrained(
            model_name, local_files_only=is_local
        ).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(
            model_name, local_files_only=is_local
        )
        self.model.eval()
        print(f"CLIPClassifierV2 initialized on device: {self.device}")
    
    def _get_class_probs(self, image: Image.Image, 
                         prompts: Dict[str, List[str]]) -> Tuple[Dict[str, float], float]:
        """
        Get class probabilities for a single image.
        Returns: (class_probs_dict, max_confidence)
        """
        class_names = list(prompts.keys())
        text_inputs = [p for cn in class_names for p in prompts[cn]]
        
        inputs = self.processor(
            text=text_inputs, images=image, 
            return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits_per_image / self.temperature
        probs = logits.softmax(dim=1)
        
        # Aggregate per-class probabilities
        num_prompts = [len(prompts[cn]) for cn in class_names]
        class_probs = {}
        start = 0
        for cn, n in zip(class_names, num_prompts):
            if n > 0:
                class_probs[cn] = probs[:, start:start+n].mean().item()
            else:
                class_probs[cn] = 0.0
            start += n
        
        max_conf = max(class_probs.values())
        return class_probs, max_conf
    
    def classify_region(self, image: Image.Image,
                        prompts: Dict[str, List[str]]) -> Tuple[str, float, Dict[str, float]]:
        """
        Classify a single cropped region.
        Returns: (predicted_class, confidence, all_class_probs)
        """
        if image is None:
            return None, 0.0, {}
        
        class_probs, conf = self._get_class_probs(image, prompts)
        
        if conf < self.confidence_threshold:
            return "uncertain", conf, class_probs
        
        pred_class = max(class_probs, key=class_probs.get)
        return pred_class, conf, class_probs
    
    def classify_with_ensemble(self, image_np: np.ndarray, 
                               mask: np.ndarray,
                               prompts: Dict[str, List[str]],
                               crop_strategies: List[str] = ['tight', 'context'],
                               overlay_strategy: str = 'none',
                               weights: List[float] = None) -> Tuple[str, float, Dict[str, float]]:
        """
        Classify using ensemble of multiple crops.
        
        Args:
            image_np: Original image
            mask: Binary mask
            prompts: Class prompts
            crop_strategies: List of crop strategies to use
            overlay_strategy: How to handle background ('none', 'blur', 'dim', 'masked')
            weights: Optional weights for each crop strategy
        
        Returns: (predicted_class, confidence, aggregated_probs)
        """
        if mask.sum() == 0:
            return None, 0.0, {}
        
        # Get crops using different strategies
        crops = []
        for strategy in crop_strategies:
            if strategy == 'tight':
                img, m = CropStrategy.tight_crop(image_np, mask, padding=10)
            elif strategy == 'context':
                img, m = CropStrategy.context_crop(image_np, mask, context_ratio=0.3)
            elif strategy == 'square':
                img, m = CropStrategy.square_crop(image_np, mask, target_size=224)
            else:
                img, m = CropStrategy.tight_crop(image_np, mask)
            
            if img is None:
                continue
            
            # Apply overlay strategy
            if overlay_strategy == 'blur':
                img = MaskOverlay.blur_background(img, m)
            elif overlay_strategy == 'dim':
                img = MaskOverlay.dim_background(img, m)
            elif overlay_strategy == 'masked':
                img = MaskOverlay.masked_only(img, m)
            elif overlay_strategy == 'boundary':
                img = MaskOverlay.highlight_boundary(img, m)
            # else 'none' - keep as is
            
            crops.append(img)
        
        if not crops:
            return None, 0.0, {}
        
        # Set weights
        if weights is None:
            weights = [1.0] * len(crops)
        weights = np.array(weights[:len(crops)])
        weights = weights / weights.sum()
        
        # Get probabilities for each crop
        all_probs = []
        for crop in crops:
            probs, _ = self._get_class_probs(crop, prompts)
            all_probs.append(probs)
        
        # Weighted average
        class_names = list(prompts.keys())
        agg_probs = {cn: 0.0 for cn in class_names}
        for probs, w in zip(all_probs, weights):
            for cn in class_names:
                agg_probs[cn] += probs.get(cn, 0) * w
        
        # Final prediction
        pred_class = max(agg_probs, key=agg_probs.get)
        confidence = agg_probs[pred_class]
        
        if confidence < self.confidence_threshold:
            return "uncertain", confidence, agg_probs
        
        return pred_class, confidence, agg_probs
    
    def classify_multiscale(self, image_np: np.ndarray,
                            mask: np.ndarray,
                            prompts: Dict[str, List[str]],
                            scales: List[float] = [0.8, 1.0, 1.2]) -> Tuple[str, float, Dict[str, float]]:
        """
        Classify using multi-scale crops (different context amounts).
        """
        crops = []
        for scale in scales:
            if scale < 1.0:
                # Tighter crop
                img, m = CropStrategy.tight_crop(image_np, mask, padding=0)
            elif scale == 1.0:
                img, m = CropStrategy.tight_crop(image_np, mask, padding=10)
            else:
                # More context
                img, m = CropStrategy.context_crop(image_np, mask, 
                                                    context_ratio=(scale - 1.0))
            
            if img is not None:
                crops.append(img)
        
        if not crops:
            return None, 0.0, {}
        
        # Average predictions across scales
        all_probs = []
        for crop in crops:
            probs, _ = self._get_class_probs(crop, prompts)
            all_probs.append(probs)
        
        class_names = list(prompts.keys())
        agg_probs = {cn: np.mean([p.get(cn, 0) for p in all_probs]) 
                     for cn in class_names}
        
        pred_class = max(agg_probs, key=agg_probs.get)
        confidence = agg_probs[pred_class]
        
        return pred_class, confidence, agg_probs


# Utility functions for backward compatibility
def crop_region_from_mask(image_np: np.ndarray, mask: np.ndarray,
                          strategy: str = 'tight',
                          overlay: str = 'none') -> Optional[Image.Image]:
    """
    Backward-compatible crop function with new options.
    
    Args:
        image_np: Input image as numpy array
        mask: Binary mask
        strategy: 'tight', 'context', 'square'
        overlay: 'none', 'blur', 'dim', 'masked', 'boundary'
    """
    if mask.sum() == 0:
        return None
    
    # Get crop
    if strategy == 'tight':
        img, m = CropStrategy.tight_crop(image_np, mask, padding=10)
    elif strategy == 'context':
        img, m = CropStrategy.context_crop(image_np, mask)
    elif strategy == 'square':
        img, m = CropStrategy.square_crop(image_np, mask)
    else:
        img, m = CropStrategy.tight_crop(image_np, mask)
    
    if img is None:
        return None
    
    # Apply overlay
    if overlay == 'blur':
        return MaskOverlay.blur_background(img, m)
    elif overlay == 'dim':
        return MaskOverlay.dim_background(img, m)
    elif overlay == 'masked':
        return MaskOverlay.masked_only(img, m)
    elif overlay == 'boundary':
        return MaskOverlay.highlight_boundary(img, m)
    else:
        return img


# Quick comparison script
if __name__ == "__main__":
    print("CLIPClassifierV2 loaded successfully!")
    print("\nAvailable crop strategies:")
    print("  - tight: Bounding box with small padding")
    print("  - context: Bounding box with 30% extra context")
    print("  - square: Square crop centered on mask centroid")
    print("\nAvailable overlay strategies:")
    print("  - none: Show full image (context visible)")
    print("  - blur: Blur background, keep mask region sharp")
    print("  - dim: Dim background to 30% brightness")
    print("  - masked: Gray background outside mask")
    print("  - boundary: Add yellow boundary around mask")
