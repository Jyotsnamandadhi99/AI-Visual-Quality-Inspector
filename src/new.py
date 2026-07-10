"""
Image Captioning using Hugging Face BLIP

Author: Jyotsna Mandadhi
"""

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import os

# =====================================
# Load Processor
# =====================================

processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

# =====================================
# Load Model
# =====================================

model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

# =====================================
# Image Path
# =====================================

image_path = r"data/raw/sample3.jpg"

image = Image.open(image_path).convert("RGB")

# =====================================
# Process Image
# =====================================

inputs = processor(
    image,
    return_tensors="pt"
)

# =====================================
# Generate Caption
# =====================================

output = model.generate(**inputs)

caption = processor.decode(
    output[0],
    skip_special_tokens=True
)

# =====================================
# Print Caption
# =====================================

print("\nGenerated Caption\n")

print(caption)

# =====================================
# Save Caption
# =====================================

os.makedirs("outputs/Captions", exist_ok=True)

with open("outputs/Captions/caption.txt", "w") as file:

    file.write("Generated Caption\n")
    file.write("=" * 40 + "\n\n")
    file.write(caption)

print("\nCaption Saved Successfully!")