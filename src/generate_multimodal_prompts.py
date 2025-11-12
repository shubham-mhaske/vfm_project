import google.generativeai as genai
import os
import json
import sys
from tqdm import tqdm
from PIL import Image
import numpy as np

try:
    # We import our own project files
    from dataset import BCSSDataset
    from clip_classification import crop_region_from_mask
except ImportError as e:
    print(f"Error: {e}")
    print("Could not import 'dataset' or 'clip_classification'.")
    print("Please make sure you are running this from the project's root directory (vfm_project).")
    sys.exit(1)


# --- Configuration ---
NUM_EXAMPLE_IMAGES = 5
# NEW FILENAME to track this change
OUTPUT_FILENAME = "configs/prompts/llm_multimodal_prompts_v1_gemini_2.5_flash.json"
CLASSES_TO_GENERATE = ['tumor', 'stroma', 'lymphocyte', 'necrosis', 'blood_vessel', 'background']
# ---------------------

def get_example_images(class_name, class_id, dataset, n_examples=5):
    """Finds N example images for a given class_id from the dataset."""
    example_images = []
    for sample in dataset:
        if len(example_images) >= n_examples:
            break
        
        unique_classes = sample['unique_classes']
        if class_id in unique_classes:
            image_np = sample['image_np']
            gt_mask = sample['mask'].numpy()
            binary_gt_mask = (gt_mask == class_id).astype(np.uint8)
            if binary_gt_mask.sum() == 0:
                continue
            cropped_pil_image = crop_region_from_mask(image_np, binary_gt_mask)
            if cropped_pil_image:
                cropped_pil_image.thumbnail((512, 512))
                example_images.append(cropped_pil_image)
                
    print(f"  Found {len(example_images)} examples for '{class_name}'")
    return example_images

def generate_multimodal_prompts():
    """
    Uses the Gemini API (multimodal) to generate visually-grounded, expert-level prompts
    for each tissue class and saves them to a JSON file.
    """
    
    try:
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    except KeyError:
        print("="*80)
        print("ERROR: GEMINI_API_KEY environment variable not set.")
        print("Please set the key with: export GEMINI_API_KEY='your-key-here'")
        print("="*80)
        sys.exit(1)

    print("Loading dataset to find example images...")
    image_dir = 'data/bcss/images'
    mask_dir = 'data/bcss/masks'
    dataset = BCSSDataset(image_dir=image_dir, mask_dir=mask_dir, split='train')
    
    # --- THIS IS THE FIX ---
    # Using 'gemini-2.5-flash-preview-09-2025' which is in your list and supports multimodal
    model = genai.GenerativeModel('gemini-2.5-flash-preview-09-2025') 
    # --- END FIX ---
    print(f"Using model: {model.model_name}")

    output_prompts = {}
    print(f"Starting 'v1 Jargon-Only' MULTIMODAL prompt generation... Saving to {OUTPUT_FILENAME}")

    for class_name in tqdm(CLASSES_TO_GENERATE, desc="Generating Prompts"):
        class_id = [k for k, v in dataset.class_names.items() if v == class_name][0]
        
        if class_name == 'background':
            print("  -> Using generic 'background' prompts.")
            output_prompts[class_name] = [
                "a blank white area", "empty slide space", "no tissue present",
                "out of focus region", "an acellular, featureless area"
            ]
            continue

        example_images = get_example_images(class_name, class_id, dataset, NUM_EXAMPLE_IMAGES)
        if not example_images:
            print(f"Warning: Could not find example images for {class_name}")
            output_prompts[class_name] = []
            continue

        # This is the "v1" meta-prompt that asks for expert jargon
        meta_prompt_text = f"""
        You are an expert computational pathologist. Look at the {len(example_images)} example images of '{class_name}' tissue.
        
        Based *only* on these images, generate 7 short, distinct, and visually specific phrases
        to describe this class for a vision-language model.

        Rules:
        1.  Focus *only* on the visual characteristics (cell shape, nucleus, color, texture, arrangement).
        2.  Each of the 7 phrases must be on its own line.
        3.  Do not use bullets or numbering (e.g., "1. ...").
        4.  Do not use generic phrases like "an image of" or "a histopathology slide of".
        """
        
        api_prompt_content = example_images + [meta_prompt_text]
        
        try:
            response = model.generate_content(api_prompt_content)
            generated_phrases = [phrase.strip() for phrase in response.text.split('\n') if phrase.strip()]
            generated_phrases = [p for p in generated_phrases if p and len(p) > 5 and not p.startswith("1.")]
            
            if not generated_phrases:
                raise Exception("LLM returned empty or invalid response.")
            
            output_prompts[class_name] = generated_phrases[:7] # Take top 7
            
        except Exception as e:
            print(f"\nError generating prompts for {class_name}: {e}")
            output_prompts[class_name] = [] # Fallback to empty list

    with open(OUTPUT_FILENAME, 'w') as f:
        json.dump(output_prompts, f, indent=4)

    print("\n" + "="*80)
    print("Multimodal 'v1 Jargon-Only' prompt generation complete!")
    print(f"Saved {len(output_prompts)} classes to {OUTPUT_FILENAME}")
    print("="*80)

if __name__ == "__main__":
    print("This script must be run after setting the GEMINI_API_KEY environment variable.")
    print("e.g., export GEMINI_API_KEY='YOUR_API_KEY_HERE'")
    generate_multimodal_prompts()