"""
Preprocessing Module

Performs:
- Resize
- Grayscale
- RGB
- Blur
- Edges

Author: Jyotsna Mandadhi
"""

import cv2
import os

image = cv2.imread(r"F:\INNOMATICS\ML\Open_Cv\AI-Visual-Quality-Inspector\data\raw\sample.jfif")

if image is None:
    print("Image not Found!")
    exit()

folders = [
    "outputs/resized",
    "outputs/grayscale",
    "outputs/rgb",
    "outputs/blur",
    "outputs/edges"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

resized = cv2.resize(image, (640,640))
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
blur = cv2.GaussianBlur(resized, (5,5), 0)
edges = cv2.Canny(gray, 100, 200)

cv2.imwrite("outputs/resized/resized.jpg", resized)
cv2.imwrite("outputs/grayscale/grayscale.jpg", gray)
cv2.imwrite("outputs/rgb/rgb.jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
cv2.imwrite("outputs/blur/blur.jpg", blur)
cv2.imwrite("outputs/edges/edges.jpg", edges)

print("Image processing completed sucessfully")