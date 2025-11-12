import google.generativeai as genai
import os
import sys

try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
except KeyError:
    print("="*80)
    print("ERROR: GEMINI_API_KEY environment variable not set.")
    print("Please set the key in your terminal before running:")
    print("export GEMINI_API_KEY='your-key-here'")
    print("="*80)
    sys.exit(1)

print("Available Gemini models:")
for m in genai.list_models():
  if 'generateContent' in m.supported_generation_methods:
    print(m.name)
