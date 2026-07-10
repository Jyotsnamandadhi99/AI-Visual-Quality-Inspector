"""
Image Captioning using Hugging Face BLIP

Author: Jyotsna Mandadhi
"""
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import os

processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

image_path = r"data/raw/sample1.jpg"

image = Image.open(image_path).convert("RGB")

inputs = processor(
    image,
    return_tensors="pt"
)

output = model.generate(**inputs)

caption = processor.decode(
    output[0],
    skip_special_tokens = True
)

print("\nGenerated Caption\n")
print(caption)

os.makedirs("outputs/Captions", exist_ok=True)
with open("outputs/Captions/caption1.txt","w") as file:
    file.write("Generated Caption\n")
    file.write("="*40+"\n\n")
    file.write(caption)
print("\nCaption Saved Successfully!")