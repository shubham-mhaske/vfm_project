import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np

def generate_text_prompts():
    """Generates an ensemble of text prompts for each class."""
    prompts = {
        'tumor': [
            "a histopathology image of a tumor",
            "cancerous tissue with malignant cells",
            "a dense cluster of large, irregularly shaped cells with dark nuclei",
            "invasive ductal carcinoma cells",
            "a region of neoplastic cells"
        ],
        'stroma': [
            "a histopathology image of stroma",
            "connective tissue supporting the tumor",
            "spindle-shaped cells with elongated nuclei",
            "fibrous tissue surrounding cancer cells",
            "a region of desmoplastic stroma"
        ],
        'lymphocyte': [
            "a histopathology image of lymphocytes",
            "a cluster of immune cells",
            "small, round cells with dark, circular nuclei and minimal cytoplasm",
            "an infiltration of lymphocytes",
            "a region of immune response in tissue"
        ],
        'necrosis': [
            "a histopathology image of necrosis",
            "dead tissue, often with fragmented cells and debris",
            "an area of cell death with loss of cell structure",
            "necrotic core of a tumor",
            "a region of eosinophilic, anucleated cells"
        ],
        'blood_vessel': [
            "a histopathology image of a blood vessel",
            "a channel for blood flow lined by endothelial cells",
            "a cross-section of a capillary or arteriole",
            "a vessel containing red blood cells",
            "vascular structure in tissue"
        ],
        'background': [
            "a histopathology image of background tissue",
            "adipose tissue or empty space",
            "fat cells and slide background",
            "a region with no distinct cellular structures",
            "normal, non-cancerous tissue"
        ]
    }
    return prompts

def crop_region_from_mask(image_np, mask):
    """Crops the region from the image corresponding to the mask."""
    if mask.sum() == 0:
        return None
    
    # Find bounding box of the mask
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Crop the image
    cropped_image_np = image_np[rmin:rmax+1, cmin:cmax+1, :]
    
    # Apply mask to the cropped image
    mask_cropped = mask[rmin:rmax+1, cmin:cmax+1]
    cropped_image_np[~mask_cropped.astype(bool)] = 255 # Set background to white
    
    return Image.fromarray(cropped_image_np)

class CLIPClassifier:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        """Initializes the CLIP classifier."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.prompts = generate_text_prompts()
        self.class_names = list(self.prompts.keys())

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
            class_probs.append(probs[:, start_index:start_index+num_prompts].mean(dim=1))
            start_index += num_prompts
        
        # Get the class with the highest average probability
        avg_probs = torch.stack(class_probs, dim=1)
        predicted_class_index = avg_probs.argmax(dim=1).item()
        
        return self.class_names[predicted_class_index]
