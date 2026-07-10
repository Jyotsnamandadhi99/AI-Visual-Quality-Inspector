from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import cv2
import os

yolo_model = YOLO("yolov8n.pt")

processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

image_path = r"data/raw/sample3.jpg"

detected_objects = []

CONFIDENCE_THRESHOLD = 0.50

results = yolo_model(image_path)

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
print("="*50)

print("AI VISUAL ANALYSIS REPORT")

print("="*50)

print()

print("Image :", os.path.basename(image_path))

print()

print("Detected Objects")

print("-"*30)

for index, (name, confidence) in enumerate(
        detected_objects,
        start=1):

    print(
        f"{index}. {name} ({confidence:.2f})"
    )

print()

print(
    f"Total Objects : {len(detected_objects)}"
)

print()

print("Image Caption")

print("-"*30)

print(caption)

print()

print("="*50)

print("Analysis Completed Successfully")

print("="*50)

os.makedirs(
    "outputs/Reports",
    exist_ok=True
)

report_path = "outputs/Reports/visual_report.txt"

with open(report_path, "w") as report:

    report.write("AI VISUAL ANALYSIS REPORT\n")

    report.write("="*50 + "\n\n")

    report.write(
        f"Image : {os.path.basename(image_path)}\n\n"
    )

    report.write("Detected Objects\n")

    report.write("-"*30 + "\n")

    for index, (name, confidence) in enumerate(
            detected_objects,
            start=1):

        report.write(
            f"{index}. {name} ({confidence:.2f})\n"
        )

    report.write("\n")

    report.write(
        f"Total Objects : {len(detected_objects)}\n\n"
    )

    report.write("Image Caption\n")

    report.write("-"*30 + "\n")

    report.write(caption)