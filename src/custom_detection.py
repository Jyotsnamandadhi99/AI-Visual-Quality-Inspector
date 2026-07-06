"""
Custom Object Detection

Author: Jyotsna Mandadhi
"""

from ultralytics import YOLO
import cv2
import os

model = YOLO("yolov8n.pt")

image_path = r"data/raw/sample3.jpg"

image = cv2.imread(image_path)

results = model(image)

CONFIDENCE_THRESHOLD = 0.50

for result in results:

    for box in result.boxes:

        confidence = float(box.conf)

        if confidence < CONFIDENCE_THRESHOLD:
            continue

        x1, y1, x2, y2 = box.xyxy[0]

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        class_id = int(box.cls)

        class_name = model.names[class_id]

        cv2.rectangle(
            image,
            (x1,y1),
            (x2,y2),
            (0,255,0),
            2
        )

        label = f"{class_name} {confidence:.2f}"

        cv2.putText(
            image,
            label,
            (x1,y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,0),
            2
        )

os.makedirs("outputs/Custom_Detections",exist_ok=True)

cv2.imwrite(
    "outputs/Custom_Detections/custom_detection.jpg",
    image
)

cv2.namedWindow("Custom Detection",cv2.WINDOW_NORMAL)

cv2.resizeWindow("Custom Detection",900,700)

cv2.imshow("Custom Detection",image)

cv2.waitKey(0)

cv2.destroyAllWindows()