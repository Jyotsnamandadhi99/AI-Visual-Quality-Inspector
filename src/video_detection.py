"""
YOLOv8 Video Object Detection

Author: Jyotsna Mandadhi
"""

from ultralytics import YOLO
import cv2
import os
import time

# ==========================================
# Load YOLO Model
# ==========================================

model = YOLO("yolov8n.pt")

# ==========================================
# Video Path
# ==========================================

video_path = r"data/raw/sample.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise FileNotFoundError("Unable to open video.")

# ==========================================
# Output Folder
# ==========================================

os.makedirs("outputs/Video_Detections", exist_ok=True)

# ==========================================
# Video Properties
# ==========================================

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    "outputs/Video_Detections/detected_video.mp4",
    fourcc,
    video_fps,
    (width, height)
)

# ==========================================
# Variables
# ==========================================

CONFIDENCE_THRESHOLD = 0.50

previous_time = time.time()

frame_number = 0

# ==========================================
# Process Video
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    object_count = 0

    results = model(frame)

    for result in results:

        for box in result.boxes:

            confidence = float(box.conf)

            if confidence < CONFIDENCE_THRESHOLD:
                continue

            object_count += 1

            class_id = int(box.cls)

            class_name = model.names[class_id]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Different Colors

            if class_name == "person":
                color = (0, 255, 0)

            elif class_name == "car":
                color = (255, 0, 0)

            elif class_name == "dog":
                color = (0, 0, 255)

            else:
                color = (0, 255, 255)

            # Bounding Box

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            # Label

            label = f"{class_name} {confidence:.2f}"

            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    # ======================================
    # FPS
    # ======================================

    current_time = time.time()

    fps = 1 / (current_time - previous_time)

    previous_time = current_time

    cv2.putText(
        frame,
        f"FPS : {fps:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 255),
        2
    )

    # ======================================
    # Frame Number
    # ======================================

    cv2.putText(
        frame,
        f"Frame : {frame_number}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2
    )

    # ======================================
    # Object Count
    # ======================================

    cv2.putText(
        frame,
        f"Objects : {object_count}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    # ======================================
    # Display
    # ======================================

    cv2.namedWindow("YOLO Video Detection", cv2.WINDOW_NORMAL)

    cv2.resizeWindow("YOLO Video Detection", 1000, 700)

    cv2.imshow("YOLO Video Detection", frame)

    # Save Video

    out.write(frame)

    # Keyboard Controls

    key = cv2.waitKey(1) & 0xFF

    # Pause

    if key == ord("p"):

        print("Video Paused. Press any key to continue...")

        cv2.waitKey(0)

    # Quit

    if key == ord("q"):
        break

# ==========================================
# Release Resources
# ==========================================

cap.release()

out.release()

cv2.destroyAllWindows()

print("\nVideo Processing Completed Successfully!")