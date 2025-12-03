#!/usr/bin/env python3
"""
Download CLIP model to cache for offline use on compute nodes.
Run this on the login node (which has internet access).
"""
import os
print("Downloading CLIP model for offline use...")

from transformers import CLIPProcessor, CLIPModel

model_name = "openai/clip-vit-base-patch32"
print(f"Downloading: {model_name}")

# This will download and cache the model
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

print(f"Model cached at: {os.environ.get('HF_HOME', '~/.cache/huggingface')}")
print("Done! Now compute nodes can use the cached model.")
