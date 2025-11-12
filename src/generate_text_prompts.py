import google.generativeai as genai
import os
import json
import sys
from tqdm import tqdm

# --- Configuration ---
# Saving to a clean path in the configs folder with a new version name
OUTPUT_FILENAME = "configs/prompts/llm_text_prompts_v1_gemini_pro_latest.json"
CLASSES_TO_GENERATE = ['tumor', 'stroma', 'lymphocyte', 'necrosis', 'blood_vessel', 'background']
# ---------------------

def generate_text_prompts_llm():
    """
    Generates visually descriptive text prompts for given classes using the stable 'gemini-pro-latest' model.
    The prompts focus on expert-level "v1 Jargon" (shape, color, texture).
    """
    print("Connecting to Google AI Studio...")
    try:
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    except KeyError:
        print("="*80)
        print("ERROR: GEMINI_API_KEY environment variable not set.")
        print("Please set the key in your terminal before running:")
        print("export GEMINI_API_KEY='your-key-here'")
        print("="*80)
        sys.exit(1)

    # --- THIS IS THE FIX ---
    # Using 'gemini-pro-latest' which is confirmed to be in your list
    model = genai.GenerativeModel('gemini-pro-latest') 
    # --- END FIX ---
    print(f"Using model: {model.model_name}")

    llm_prompts = {}
    print(f"Generating 'v1 Jargon-Only' prompts... Saving to {OUTPUT_FILENAME}")
    for class_name in tqdm(CLASSES_TO_GENERATE, desc="Generating Prompts"):
        
        # This is the "v1" meta-prompt that asks for expert jargon
        prompt_text = (
            f"You are an expert computational pathologist. I need to generate text prompts for a\n"
            f"CLIP vision-language model to classify breast cancer histology images.\n\n"
            f"For the tissue class '{class_name}', please generate 7 short, descriptive, \n"
            f"and visually specific phrases.\n\n"
            f"Rules:\n"
            f"1.  Focus *only* on the visual characteristics (cell shape, nucleus, color, texture, arrangement).\n"
            f"2.  Each of the 7 phrases must be on its own line.\n"
            f"3.  Do not use bullets, numbering, or JSON formatting. Just plain text, one line per phrase.\n"
            f"4.  Do not use generic phrases like 'a photo of' or 'an image of'."
        )
        
        try:
            response = model.generate_content(prompt_text)
            
            # Process the plain text response
            generated_phrases = [phrase.strip() for phrase in response.text.split('\n') if phrase.strip()]
            generated_phrases = [p for p in generated_phrases if p and len(p) > 5 and not p.startswith("1.")]
            
            if not generated_phrases:
                raise Exception("LLM returned empty or invalid response.")
                
            llm_prompts[class_name] = generated_phrases[:7] # Take the top 7
            
        except Exception as e:
            print(f"\nWarning: Could not generate prompts for class {class_name}. Error: {e}")
            llm_prompts[class_name] = [] # Fallback to empty list

    return llm_prompts

if __name__ == "__main__":
    print("Generating 'v1 Jargon-Only' text prompts using Gemini...")
    prompts = generate_text_prompts_llm()
    with open(OUTPUT_FILENAME, 'w') as f:
        json.dump(prompts, f, indent=4)
    print("\n" + "="*80)
    print(f"Generated LLM text prompts saved to {OUTPUT_FILENAME}")
    print("="*80)