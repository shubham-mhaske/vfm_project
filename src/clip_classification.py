import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
import json
import os
import sys

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
    def __init__(self, prompt_file_path: str, model_name="openai/clip-vit-base-patch32"):
        """
        Initializes the CLIP classifier by loading prompts from a specified JSON file.
        
        Args:
            prompt_file_path (str): Path to the JSON file containing prompt ensembles.
            model_name (str): The CLIP model to load from Hugging Face.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name, use_fast=True)
        
        # Load prompts from the specified JSON file
        self.prompts = load_prompts_from_json(prompt_file_path) 
        self.class_names = list(self.prompts.keys())
        print(f"CLIPClassifier initialized with prompts from: {prompt_file_path}")

    def classify_region(self, image):
        """Classifies a cropped image region using CLIP."""
        if image is None:
            return None

        # Create text prompts for all classes
        text_inputs = [prompt for class_name in self.class_names for prompt in self.prompts[class_name]]
        
        # Process image and text
        inputs = self.processor(text=text_inputs, images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Calculate similarity
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)

        # Average the probabilities for each class ensemble
        num_prompts_per_class = [len(self.prompts[class_name]) for class_name in self.class_names]
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
        
        return self.class_names[predicted_class_index]