import google.generativeai as genai
import os
import json
import sys
from tqdm import tqdm

OUTPUT_FILENAME = "configs/prompts/llm_text_prompts_v3_fewshot.json"
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

def generate_fewshot_text_prompts():
    """
    Uses few-shot learning with successful v2 prompts to generate new prompts.
    """
    print("Connecting to Google AI Studio (Gemini)...")
    try:
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    except KeyError:
        print("="*80)
        print("ERROR: GEMINI_API_KEY environment variable not set.")
        print("Please set the key: export GEMINI_API_KEY='your-key-here'")
        print("="*80)
        sys.exit(1)

    model = genai.GenerativeModel('gemini-pro-latest')
    print(f"Using model: {model.model_name}")
    
    llm_prompts = {}
    print(f"\nGenerating 'v3 Few-Shot' prompts... Saving to {OUTPUT_FILENAME}")
    print("Using successful hardcoded v2 prompts as examples.\n")

    for class_name in tqdm(CLASSES_TO_GENERATE, desc="Generating Prompts"):
        examples = SUCCESSFUL_EXAMPLES[class_name]
        
        # Build contrastive examples from other classes
        other_classes_examples = []
        for other_class in CLASSES_TO_GENERATE:
            if other_class != class_name:
                other_classes_examples.append(
                    f"{other_class}: {SUCCESSFUL_EXAMPLES[other_class][0]}"
                )
        
        prompt_text = f"""You are creating CLIP-friendly prompts for H&E breast cancer histology classification.

TASK: Generate 5 NEW prompts for the class '{class_name}' that follow the EXACT STYLE and SPECIFICITY of these PROVEN SUCCESSFUL examples:

SUCCESSFUL EXAMPLES for '{class_name}':
1. "{examples[0]}"
2. "{examples[1]}"
3. "{examples[2]}"

CRITICAL RULES (learned from successful prompts):
1. Use SPECIFIC color/size/texture combinations that are UNIQUE to this class
2. Emphasize DISTINCTIVE features that separate this class from others
3. Use simple, everyday language (no medical jargon)
4. Keep the same level of detail and specificity as the examples

WHAT MAKES THESE EXAMPLES WORK:
- They use CONCRETE visual descriptors (colors, sizes, shapes, textures)
- They combine multiple features that TOGETHER uniquely identify the class
- They avoid generic terms that could apply to multiple tissue types

CONTRAST WITH OTHER CLASSES (what NOT to match):
{chr(10).join(other_classes_examples)}

CONSTRAINTS:
- Each prompt must be ONE concise sentence
- NO bullet points, numbering, or prefixes like "an image of"
- Must be DISTINCT from the other class prompts listed above
- Focus on what makes '{class_name}' visually DIFFERENT

Generate 5 NEW prompts in this exact style:"""

        try:
            response = model.generate_content(prompt_text)
            generated_phrases = []
            
            # Parse response - handle both numbered and plain text
            for line in response.text.split('\n'):
                line = line.strip()
                # Remove numbering, bullets, quotes
                line = line.lstrip('0123456789.-) "\'')
                line = line.rstrip('"\'')
                if line and len(line) > 10 and not line.startswith("Here") and not line.startswith("These"):
                    generated_phrases.append(line)
            
            if len(generated_phrases) < 3:
                print(f"\n⚠️  Warning: Only {len(generated_phrases)} prompts generated for {class_name}, using examples as fallback")
                generated_phrases = examples[:5]
            
            # Keep original examples + new variations
            llm_prompts[class_name] = examples + generated_phrases[:5]
            llm_prompts[class_name] = llm_prompts[class_name][:7]  # Max 7 total
            
        except Exception as e:
            print(f"\n❌ Error generating prompts for {class_name}: {e}")
            print("   Using original successful examples as fallback.")
            llm_prompts[class_name] = examples[:5]

    return llm_prompts

if __name__ == "__main__":
    print("="*80)
    print("FEW-SHOT LLM PROMPT GENERATION (v3)")
    print("="*80)
    print("\nStrategy: Use proven successful prompts as examples to guide LLM")
    print("Expected: Better class discrimination and higher accuracy than v2\n")
    
    prompts = generate_fewshot_text_prompts()
    
    os.makedirs("configs/prompts", exist_ok=True)
    with open(OUTPUT_FILENAME, 'w') as f:
        json.dump(prompts, f, indent=4)
    
    print("\n" + "="*80)
    print("✅ Few-shot v3 prompts generated successfully!")
    print(f"📁 Saved to: {OUTPUT_FILENAME}")
    print("\nNext steps:")
    print("  1. Run evaluation:")
    print("     python src/evaluation.py \\")
    print("       --clip_prompts configs/prompts/llm_text_prompts_v3_fewshot.json \\")
    print("       --output_dir results/exp3_v3_llm_fewshot")
    print("\n  2. Compare to hardcoded v2 (40.4% baseline)")
    print("="*80)
