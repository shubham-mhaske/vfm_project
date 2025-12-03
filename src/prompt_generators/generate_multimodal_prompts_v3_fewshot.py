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

NUM_EXAMPLE_IMAGES = 3  # Reduced to focus on quality
OUTPUT_FILENAME = "configs/prompts/llm_multimodal_prompts_v3_fewshot.json"
CLASSES_TO_GENERATE = ['tumor', 'stroma', 'lymphocyte', 'necrosis', 'blood_vessel']

# Successful prompts from hardcoded v2 (40.4% accuracy) to use as examples
SUCCESSFUL_EXAMPLES = {
    "tumor": [
        "densely crowded dark purple cells packed together",
        "large irregular purple nuclei in chaotic arrangement",
        "thick masses of deep purple tissue with high cell density"
    ],
    "stroma": [
        "bright pink wavy fibers forming streaming patterns",
        "light pink collagen with scattered thin elongated nuclei",
        "parallel bundles of pink fibrous tissue"
    ],
    "lymphocyte": [
        "many tiny dark blue dots scattered throughout",
        "small uniform round purple circles densely clustered",
        "dense groups of tiny dark blue spheres"
    ],
    "necrosis": [
        "pale ghostly pink tissue with faded blurry appearance",
        "washed out pale areas with fragmented cell debris",
        "smudgy faint pink tissue losing structure"
    ],
    "blood_vessel": [
        "circular empty white space surrounded by thin pink wall",
        "ring-shaped hollow area with pink rim",
        "round opening with thin pink tissue lining the edge"
    ]
}

def get_example_images(class_name, class_id, dataset, n_examples=3):
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

def generate_fewshot_multimodal_prompts():
    """
    Uses few-shot learning with successful v2 prompts + images to generate new prompts.
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
    print(f"\nStarting 'v3 Few-Shot MULTIMODAL' prompt generation...")
    print("Using successful hardcoded v2 prompts + images as examples.\n")

    for class_name in tqdm(CLASSES_TO_GENERATE, desc="Generating Prompts"):
        class_id = [k for k, v in dataset.class_names.items() if v == class_name][0]
        
        example_images = get_example_images(class_name, class_id, dataset, NUM_EXAMPLE_IMAGES)
        if not example_images:
            print(f"⚠️  Warning: No examples found for {class_name}, using text-only fallback")
            output_prompts[class_name] = SUCCESSFUL_EXAMPLES[class_name][:5]
            continue

        examples = SUCCESSFUL_EXAMPLES[class_name]
        
        # Build contrastive examples
        other_classes_examples = []
        for other_class in CLASSES_TO_GENERATE:
            if other_class != class_name:
                other_classes_examples.append(
                    f"  - {other_class}: {SUCCESSFUL_EXAMPLES[other_class][0]}"
                )

        meta_prompt_text = f"""You are creating CLIP-friendly prompts for H&E histology classification.

Look at these {len(example_images)} example images of '{class_name}' tissue.

PROVEN SUCCESSFUL PROMPTS for '{class_name}' (40.4% accuracy):
1. "{examples[0]}"
2. "{examples[1]}"
3. "{examples[2]}"

TASK: Generate 5 NEW prompts that:
1. Follow the EXACT STYLE of the successful examples above
2. Describe what you SEE in the images using SIMPLE color/shape/texture words
3. Match the SPECIFICITY and CONCRETENESS of the examples
4. Are DISTINCT from other tissue classes

WHAT MAKES THESE EXAMPLES WORK:
✅ Specific color emphasis (dark purple, bright pink, pale pink, tiny blue dots)
✅ Size descriptors (large, tiny, small, thick, thin)
✅ Texture/arrangement (crowded, scattered, wavy, sparse, dense)
✅ Combinations that UNIQUELY identify this class

AVOID (lessons from failed v2):
❌ Architectural language ("branching patterns", "lace-like")
❌ Generic descriptors that apply to multiple classes
❌ Complex spatial descriptions CLIP can't understand

CONTRAST WITH OTHER CLASSES (what NOT to describe):
{chr(10).join(other_classes_examples)}

CONSTRAINTS:
- ONE concise sentence per prompt
- NO bullets, numbering, or "an image of" prefixes
- Use EVERYDAY words, not medical jargon
- Focus on DOMINANT visual features in the images

Look at the images and generate 5 prompts in the style of the successful examples:"""
        
        api_prompt_content = example_images + [meta_prompt_text]
        
        try:
            response = model.generate_content(api_prompt_content)
            generated_phrases = []
            
            for line in response.text.split('\n'):
                line = line.strip()
                line = line.lstrip('0123456789.-) "\'')
                line = line.rstrip('"\'')
                if line and len(line) > 10 and not line.startswith("Here") and not line.startswith("These"):
                    generated_phrases.append(line)
            
            if len(generated_phrases) < 3:
                print(f"\n⚠️  Warning: Only {len(generated_phrases)} prompts for {class_name}, using fallback")
                generated_phrases = examples[:5]
            
            # Keep original examples + new variations
            output_prompts[class_name] = examples + generated_phrases[:5]
            output_prompts[class_name] = output_prompts[class_name][:7]
            
        except Exception as e:
            print(f"\n❌ Error generating prompts for {class_name}: {e}")
            print("   Using original successful examples as fallback.")
            output_prompts[class_name] = examples[:5]

    os.makedirs("configs/prompts", exist_ok=True)
    with open(OUTPUT_FILENAME, 'w') as f:
        json.dump(output_prompts, f, indent=4)
    
    print("\n" + "="*80)
    print("✅ Few-shot multimodal v3 prompts generated!")
    print(f"📁 Saved to: {OUTPUT_FILENAME}")
    print("\nNext steps:")
    print("  1. Run evaluation:")
    print("     python src/evaluation.py \\")
    print("       --clip_prompts configs/prompts/llm_multimodal_prompts_v3_fewshot.json \\")
    print("       --output_dir results/exp5_v3_llm_multimodal_fewshot")
    print("\n  2. Compare to hardcoded v2 (40.4% baseline)")
    print("="*80)

if __name__ == "__main__":
    print("="*80)
    print("FEW-SHOT MULTIMODAL LLM PROMPT GENERATION (v3)")
    print("="*80)
    print("\nStrategy: Use images + proven prompts to guide multimodal LLM")
    print("Expected: Avoid v2 architectural language trap, maintain specificity\n")
    generate_fewshot_multimodal_prompts()
