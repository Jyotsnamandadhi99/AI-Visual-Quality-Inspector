"""
Image Transformation Module

Performs:
- Rotation
- Flipping
- Scaling
- Translation
- ROI Selection

Author: Jyotsna Mandadhi
"""

import cv2
import os
import numpy as np

image = cv2.imread(r"F:\INNOMATICS\ML\Open_Cv\AI-Visual-Quality-Inspector\data\raw\sample.jfif")

if image is None:
    raise FileNotFoundError("Image not found!")

OUTPUT_FOLDERS = [
    "outputs/rotated",
    "outputs/flipped",
    "outputs/cropped",
    "outputs/translated",
    "outputs/scaled",
    "outputs/roi",
]

for folder in OUTPUT_FOLDERS:
    os.makedirs(folder, exist_ok=True)

# Rotation

height, width = image.shape[:2]

center = (width//2, height//2)

rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1)

rotated = cv2.warpAffine(image, rotation_matrix, (width, height))

cv2.imwrite("outputs/rotated/rotated.jpg", rotated)

# Flip

horizontal = cv2.flip(image, 1)
vertical = cv2.flip(image, 0)

cv2.imwrite("outputs/flipped/horizontal.jpg", horizontal)
cv2.imwrite("outputs/flipped/vertical.jpg", vertical)

# Crop

ROI_X1 = 200
ROI_Y1 = 150
ROI_X2 = 700
ROI_Y2 = 650

cropped = image[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]

cv2.imwrite("outputs/cropped/cropped.jpg", cropped)

# Region of Interest Rectangle

roi_image = image.copy()

cv2.rectangle(
    roi_image,
    (200,150),
    (700,650),
    (0,255,0),
    3
)

cv2.imwrite("outputs/roi/roi.jpg", roi_image)

# Move Image

translation_matrix = np.float32([
    [1,0,100],
    [0,1,50]
])

translated = cv2.warpAffine(
    image,
    translation_matrix,
    (width,height)
)

cv2.imwrite("outputs/translated/translated.jpg", translated)

# Scale

scaled = cv2.resize(
    image,
    None,
    fx=0.5,
    fy=0.5
)

cv2.imwrite("outputs/scaled/scaled.jpg", scaled)