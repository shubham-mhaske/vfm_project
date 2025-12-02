import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
import json
import os
import sys

# Default local path for offline CLIP model on HPRC
DEFAULT_CLIP_LOCAL_PATH = os.environ.get('CLIP_MODEL_PATH', None)


def get_clip_model_path():
    """Get CLIP model path - local if available, otherwise HuggingFace."""
    # Check environment variable first
    if DEFAULT_CLIP_LOCAL_PATH and os.path.exists(DEFAULT_CLIP_LOCAL_PATH):
        return DEFAULT_CLIP_LOCAL_PATH
    
    # Check common local paths on HPRC
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
    
    # Fall back to HuggingFace (requires internet)
    return "openai/clip-vit-base-patch32"


def load_prompts_from_json(json_path: str):
    """Loads a prompt dictionary from a JSON file."""
    if not os.path.exists(json_path):
        print("="*80)
        print(f"ERROR: Prompt file not found: {json_path}")
        print("This file must be provided to the CLIPClassifier.")
        print("="*80)
        raise FileNotFoundError(json_path)
            
    with open(json_path, 'r') as f:
        prompts = json.load(f)
    return prompts

def crop_region_from_mask(image_np, mask):
    """Crops the region from the image corresponding to the mask."""
    if mask.sum() == 0:
        return None
    
    # Find bounding box of the mask
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    # Handle empty mask edge case
    if not np.any(rows) or not np.any(cols):
        return None
        
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Crop the image
    cropped_image_np = image_np[rmin:rmax+1, cmin:cmax+1, :]
    
    # Apply mask to the cropped image
    mask_cropped = mask[rmin:rmax+1, cmin:cmax+1]
    cropped_image_np[~mask_cropped.astype(bool)] = 255 # Set background to white
    
    return Image.fromarray(cropped_image_np)

class CLIPClassifier:
    def __init__(self, model_name=None, device=None):
        """
        Initializes the CLIP classifier.
        
        Args:
            model_name (str): The CLIP model to load. Can be:
                - HuggingFace model name (e.g., "openai/clip-vit-base-patch32")
                - Local path to downloaded model
                - None (auto-detect local or use HuggingFace)
            device (torch.device, optional): The device to load the model on. Defaults to CPU.
        """
        self.device = device if device else torch.device("cpu")
        
        # Auto-detect model path if not specified
        if model_name is None:
            model_name = get_clip_model_path()
        
        print(f"Loading CLIP model from: {model_name}")
        self.model = CLIPModel.from_pretrained(model_name, local_files_only=os.path.exists(model_name)).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name, local_files_only=os.path.exists(model_name))
        print(f"CLIPClassifier initialized on device: {self.device}")

    def classify_region(self, image, prompts):
        """Classifies a cropped image region using CLIP with a given set of prompts."""
        if image is None:
            return None

        class_names = list(prompts.keys())
        # Create text prompts for all classes
        text_inputs = [prompt for class_name in class_names for prompt in prompts[class_name]]
        
        # Process image and text
        inputs = self.processor(text=text_inputs, images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Calculate similarity
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)

        # Average the probabilities for each class ensemble
        num_prompts_per_class = [len(prompts[class_name]) for class_name in class_names]
        class_probs = []
        start_index = 0
        for num_prompts in num_prompts_per_class:
            # Handle cases where a class might have 0 prompts
            if num_prompts == 0:
                class_probs.append(torch.tensor(0.0).to(self.device))
                continue
            
            class_probs.append(probs[:, start_index:start_index+num_prompts].mean(dim=1))
            start_index += num_prompts
        
        # Get the class with the highest average probability
        avg_probs = torch.stack(class_probs, dim=1)
        predicted_class_index = avg_probs.argmax(dim=1).item()
        
        return class_names[predicted_class_index]