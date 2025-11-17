import google.generativeai as genai
import os
import json
import sys
from tqdm import tqdm

OUTPUT_FILENAME = "configs/prompts/llm_text_prompts_v2_clip_friendly.json"
CLASSES_TO_GENERATE = ['tumor', 'stroma', 'lymphocyte', 'necrosis', 'blood_vessel']

def generate_text_prompts_llm_v2():
    """
    Generates CLIP-friendly, research-informed text prompts using simple visual language.
    """
    print("Connecting to Google AI Studio...")
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
    print(f"Generating 'v2 CLIP-Friendly' prompts... Saving to {OUTPUT_FILENAME}")

    for class_name in tqdm(CLASSES_TO_GENERATE, desc="Generating Prompts"):
        prompt_text = (
            f"You are helping create text descriptions for a CLIP vision-language model to classify "
            f"H&E-stained breast cancer histology images. CLIP was trained on natural images and everyday language, "
            f"NOT medical textbooks.\n\n"
            f"For the tissue class '{class_name}', generate 7 SHORT, visually descriptive phrases.\n\n"
            f"CRITICAL RULES:\n"
            f"1. Use SIMPLE, EVERYDAY language (avoid medical jargon like 'pleomorphic', 'hyperchromatic', 'desmoplastic').\n"
            f"2. Emphasize COLORS from H&E staining: dark purple/blue (nuclei), bright pink (collagen/stroma), pale pink (necrosis).\n"
            f"3. Emphasize SIZE and SHAPE: 'tiny dots', 'large irregular', 'wavy fibers', 'circular hollow'.\n"
            f"4. Emphasize TEXTURE and ARRANGEMENT: 'densely packed', 'scattered', 'crowded', 'sparse', 'wavy patterns'.\n"
            f"5. Each phrase must be one line, no bullets, numbering, or 'an image of' prefix.\n"
            f"6. Make phrases DISTINCT from other tissue types.\n\n"
            f"EXAMPLES OF GOOD PROMPTS:\n"
            f"- 'densely crowded dark purple cells packed together'\n"
            f"- 'bright pink wavy fibers forming streaming patterns'\n"
            f"- 'many tiny dark blue dots scattered throughout'\n"
            f"- 'pale ghostly pink tissue with faded blurry appearance'\n"
            f"- 'circular empty white space surrounded by thin pink wall'\n\n"
            f"EXAMPLES OF BAD PROMPTS:\n"
            f"- 'infiltrating nests of pleomorphic epithelial cells'  ❌\n"
            f"- 'desmoplastic stroma with hyalinized collagen'  ❌\n\n"
            f"Now generate 7 SIMPLE, COLOR-FOCUSED phrases for '{class_name}':"
        )

        try:
            response = model.generate_content(prompt_text)
            generated_phrases = [phrase.strip() for phrase in response.text.split('\n') if phrase.strip()]
            generated_phrases = [p for p in generated_phrases if p and len(p) > 5 and not p.startswith("1.") and not p.startswith("-")]
            
            if not generated_phrases:
                raise Exception("LLM returned empty response.")
            
            llm_prompts[class_name] = generated_phrases[:7]
        except Exception as e:
            print(f"\nWarning: Could not generate prompts for {class_name}. Error: {e}")
            llm_prompts[class_name] = []

    return llm_prompts

if __name__ == "__main__":
    print("Generating 'v2 CLIP-Friendly' text prompts...")
    prompts = generate_text_prompts_llm_v2()
    
    os.makedirs("configs/prompts", exist_ok=True)
    with open(OUTPUT_FILENAME, 'w') as f:
        json.dump(prompts, f, indent=4)
    
    print("\n" + "="*80)
    print(f"✅ Generated v2 prompts saved to {OUTPUT_FILENAME}")
    print("="*80)
