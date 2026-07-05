"""
YOLOv8 Object Detection

Author: Jyotsna Mandadhi
"""

from ultralytics import YOLO
import cv2
import os

# ======================================
# Load YOLO Model
# ======================================

model = YOLO("yolov8n.pt")

# ======================================
# Image Path
# ======================================

image_path = r"data\raw\sample4.jpg"

# Run Object Detection
results = model(image_path)

# ======================================
# Create Output Folder
# ======================================

os.makedirs("outputs/Detections", exist_ok=True)

# Confidence Threshold
CONFIDENCE_THRESHOLD = 0.50

# Dictionary to Count Objects
object_count = {}

# ======================================
# Display and Save Detection
# ======================================

for result in results:

    # Draw Bounding Boxes
    annotated = result.plot()

    cv2.namedWindow("YOLO Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLO Detection", 900, 700)

    cv2.imshow("YOLO Detection", annotated) 

    # Save Original Quality Image
    image_name = os.path.basename(image_path)

    output_path = os.path.join(
        "outputs/Detections",
        f"detected_{image_name}"
    )

    cv2.imwrite(output_path, annotated)

    print("Detection Image Saved Successfully!")

    cv2.waitKey(0)

cv2.destroyAllWindows()

# ======================================
# Print Detected Objects
# ======================================

print("\nDetected Objects\n")

for result in results:

    for box in result.boxes:

        class_id = int(box.cls)

        confidence = float(box.conf)

        class_name = model.names[class_id]

        # Confidence Threshold
        if confidence >= CONFIDENCE_THRESHOLD:

            print(f"{class_name:<15} Confidence : {confidence:.2f}")

            # Count Objects
            object_count[class_name] = (
                object_count.get(class_name, 0) + 1
            )

# ======================================
# Print Object Count
# ======================================

print("\nObject Count\n")

for name, count in object_count.items():

    print(f"{name} : {count}")

# ======================================
# Generate Report
# ======================================

report_path = "outputs/Detections/report.txt"

with open(report_path, "w") as report:

    report.write("YOLO Detection Report\n")
    report.write("=" * 40 + "\n\n")

    report.write(f"Image Name : {image_name}\n\n")

    report.write("Confidence Threshold : ")
    report.write(f"{CONFIDENCE_THRESHOLD}\n\n")

    report.write("Detected Objects\n")
    report.write("-" * 25 + "\n")

    for result in results:

        for box in result.boxes:

            class_id = int(box.cls)

            confidence = float(box.conf)

            class_name = model.names[class_id]

            if confidence >= CONFIDENCE_THRESHOLD:

                report.write(
                    f"{class_name} : {confidence:.2f}\n"
                )

    report.write("\n")

    report.write("Object Count\n")
    report.write("-" * 25 + "\n")

    for name, count in object_count.items():

        report.write(f"{name} : {count}\n")

print("\nReport Generated Successfully!")