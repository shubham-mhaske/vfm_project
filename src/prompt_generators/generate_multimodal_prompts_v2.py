import google.generativeai as genai
import os
import json
import sys
from tqdm import tqdm
from PIL import Image
import numpy as np

try:
    from dataset import BCSSDataset
    from clip_classification import crop_region_from_mask
except ImportError as e:
    print(f"Error: {e}")
    print("Run from project root directory (vfm_project).")
    sys.exit(1)

NUM_EXAMPLE_IMAGES = 5
OUTPUT_FILENAME = "configs/prompts/llm_multimodal_prompts_v2_clip_friendly.json"
CLASSES_TO_GENERATE = ['tumor', 'stroma', 'lymphocyte', 'necrosis', 'blood_vessel']

def get_example_images(class_name, class_id, dataset, n_examples=5):
    """Finds N example images for a given class_id."""
    example_images = []
    for sample in dataset:
        if len(example_images) >= n_examples:
            break
        if class_id in sample['unique_classes']:
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

def generate_multimodal_prompts_v2():
    """
    Uses Gemini API (multimodal) to generate CLIP-friendly, simple visual prompts.
    """
    try:
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    except KeyError:
        print("="*80)
        print("ERROR: GEMINI_API_KEY not set.")
        print("Set with: export GEMINI_API_KEY='your-key-here'")
        print("="*80)
        sys.exit(1)

    print("Loading dataset to find example images...")
    dataset = BCSSDataset(image_dir='data/bcss/images', mask_dir='data/bcss/masks', split='train')
    
    model = genai.GenerativeModel('gemini-2.5-flash-preview-09-2025')
    print(f"Using model: {model.model_name}")
    
    output_prompts = {}
    print(f"Starting 'v2 CLIP-Friendly' MULTIMODAL prompt generation...")

    for class_name in tqdm(CLASSES_TO_GENERATE, desc="Generating Prompts"):
        class_id = [k for k, v in dataset.class_names.items() if v == class_name][0]
        
        example_images = get_example_images(class_name, class_id, dataset, NUM_EXAMPLE_IMAGES)
        if not example_images:
            print(f"Warning: No examples found for {class_name}")
            output_prompts[class_name] = []
            continue

        meta_prompt_text = f"""
You are helping create visual descriptions for a CLIP vision-language model. CLIP was trained on everyday images and simple language, NOT medical literature.

Look at these {len(example_images)} example images of '{class_name}' tissue (H&E staining). Generate 7 SHORT phrases describing what you SEE.

CRITICAL RULES:
1. Use SIMPLE, EVERYDAY words a non-expert would understand.
2. Describe COLORS you see: dark purple, bright pink, pale pink, blue dots, white spaces.
3. Describe SIZES and SHAPES: tiny, large, round, irregular, wavy, circular.
4. Describe TEXTURES: crowded, scattered, dense, sparse, smooth, fibrous.
5. NO medical jargon (avoid: pleomorphic, hyperchromatic, desmoplastic, karyorrhectic).
6. Each phrase on one line, no bullets or numbering.
7. Focus on what makes THIS class visually DIFFERENT from others.

EXAMPLES OF GOOD DESCRIPTIONS:
- "densely packed dark purple cells with minimal space"
- "bright pink wavy fibers running in patterns"
- "tiny uniform dark blue dots clustered together"
- "pale washed out pink areas with blurry edges"
- "circular hollow space with thin pink rim"

EXAMPLES OF BAD DESCRIPTIONS:
- "infiltrating nests of atypical cells"  ❌
- "hypocellular fibrous stroma"  ❌

Look at the images and describe what a person with normal vision would notice about colors, shapes, and patterns.
"""
        
        api_prompt_content = example_images + [meta_prompt_text]
        
        try:
            response = model.generate_content(api_prompt_content)
            generated_phrases = [phrase.strip() for phrase in response.text.split('\n') if phrase.strip()]
            generated_phrases = [p for p in generated_phrases if p and len(p) > 5 and not p.startswith("1.") and not p.startswith("-")]
            
            if not generated_phrases:
                raise Exception("LLM returned empty response.")
            
            output_prompts[class_name] = generated_phrases[:7]
        except Exception as e:
            print(f"\nError generating prompts for {class_name}: {e}")
            output_prompts[class_name] = []

    os.makedirs("configs/prompts", exist_ok=True)
    with open(OUTPUT_FILENAME, 'w') as f:
        json.dump(output_prompts, f, indent=4)
    
    print("\n" + "="*80)
    print("✅ Multimodal 'v2 CLIP-Friendly' prompt generation complete!")
    print(f"Saved to {OUTPUT_FILENAME}")
    print("="*80)

if __name__ == "__main__":
    generate_multimodal_prompts_v2()
