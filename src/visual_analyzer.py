"""
AI Visual Analyzer
Day 9

Author: Jyotsna Mandadhi
"""

# ==========================================
# Import Libraries
# ==========================================

from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from datetime import datetime
import os

# ==========================================
# Load YOLO Model
# ==========================================

yolo_model = YOLO("yolov8n.pt")

# ==========================================
# Load BLIP Model
# ==========================================

processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

# ==========================================
# Image Path
# ==========================================

image_path = r"data/raw/sample3.jpg"

# ==========================================
# YOLO Object Detection
# ==========================================

CONFIDENCE_THRESHOLD = 0.50

results = yolo_model(image_path)

detected_objects = []

unique_objects = set()

for result in results:

    for box in result.boxes:

        confidence = float(box.conf)

        if confidence < CONFIDENCE_THRESHOLD:
            continue

        class_id = int(box.cls)

        class_name = yolo_model.names[class_id]

        detected_objects.append(
            (class_name, confidence)
        )

        unique_objects.add(class_name)

# ==========================================
# Most Confident Detection
# ==========================================

most_confident = max(
    detected_objects,
    key=lambda x: x[1]
)

# ==========================================
# BLIP Image Captioning
# ==========================================

image = Image.open(image_path).convert("RGB")

inputs = processor(
    image,
    return_tensors="pt"
)

output = blip_model.generate(
    **inputs,
    max_new_tokens=30
)

caption = processor.decode(
    output[0],
    skip_special_tokens=True
)

caption = caption.capitalize()

if not caption.endswith("."):
    caption += "."

# ==========================================
# Console Report
# ==========================================

print("=" * 60)

print("AI VISUAL ANALYSIS REPORT")

print("=" * 60)

print()

print("Date & Time :", datetime.now())

print()

print("Image :", os.path.basename(image_path))

print()

print("Detected Objects")

print("-" * 40)

for index, (name, confidence) in enumerate(
        detected_objects,
        start=1):

    print(
        f"{index}. {name:<15} {confidence:.2f}"
    )

print()

print(
    f"Total Objects : {len(detected_objects)}"
)

print()

print("Unique Objects")

print("-" * 40)

for obj in unique_objects:

    print(obj)

print()

print("Most Confident Detection")

print("-" * 40)

print(
    f"{most_confident[0]} ({most_confident[1]:.2f})"
)

print()

print("Image Caption")

print("-" * 40)

print(caption)

print()

print("=" * 60)

print("Analysis Completed Successfully")

print("=" * 60)

# ==========================================
# Save Report
# ==========================================

os.makedirs(
    "outputs/Reports",
    exist_ok=True
)

report_path = "outputs/Reports/visual_report.txt"

with open(report_path, "w") as report:

    report.write("=" * 60 + "\n")
    report.write("AI VISUAL ANALYSIS REPORT\n")
    report.write("=" * 60 + "\n\n")

    report.write(
        f"Date & Time : {datetime.now()}\n\n"
    )

    report.write(
        f"Image : {os.path.basename(image_path)}\n\n"
    )

    report.write("Detected Objects\n")
    report.write("-" * 40 + "\n")

    for index, (name, confidence) in enumerate(
            detected_objects,
            start=1):

        report.write(
            f"{index}. {name:<15} {confidence:.2f}\n"
        )

    report.write("\n")

    report.write(
        f"Total Objects : {len(detected_objects)}\n\n"
    )

    report.write("Unique Objects\n")
    report.write("-" * 40 + "\n")

    for obj in unique_objects:

        report.write(f"{obj}\n")

    report.write("\n")

    report.write("Most Confident Detection\n")
    report.write("-" * 40 + "\n")

    report.write(
        f"{most_confident[0]} ({most_confident[1]:.2f})\n\n"
    )

    report.write("Image Caption\n")
    report.write("-" * 40 + "\n")

    report.write(caption + "\n\n")

    report.write("Summary\n")
    report.write("-" * 40 + "\n")

    report.write(
        f"YOLO detected {len(detected_objects)} object(s).\n"
    )

    report.write(
        f"BLIP described the image as:\n"
    )

    report.write(
        f"\"{caption}\"\n"
    )

print("\nReport Saved Successfully!")