"""
Bottle Dataset Preprocessing

Author : Jyotsna Mandadhi
Project : AI Visual Quality Inspector
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# PATHS
# ==========================================================

TRAIN_FOLDER = r"data/raw/bottle/train/good"

OUTPUT_FOLDER = r"model_outputs/preprocessing"

PROCESSED_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "processed_images"
)

COMPARISON_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "comparison"
)

HISTOGRAM_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "histograms"
)

SAMPLE_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "samples"
)

os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(COMPARISON_FOLDER, exist_ok=True)
os.makedirs(HISTOGRAM_FOLDER, exist_ok=True)
os.makedirs(SAMPLE_FOLDER, exist_ok=True)

# ==========================================================
# PARAMETERS
# ==========================================================

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256

# ==========================================================
# IMAGE LIST
# ==========================================================

image_files = sorted(os.listdir(TRAIN_FOLDER))

statistics = []

processed_count = 0

comparison_saved = False
histogram_saved = False
sample_saved = False

print("=" * 60)
print("Bottle Dataset Preprocessing")
print("=" * 60)

# ==========================================================
# PROCESS ALL IMAGES
# ==========================================================

for image_name in image_files:

    image_path = os.path.join(
        TRAIN_FOLDER,
        image_name
    )

    image = cv2.imread(image_path)

    if image is None:
        continue

    # ------------------------------------------------------
    # Resize
    # ------------------------------------------------------

    resized = cv2.resize(
        image,
        (IMAGE_WIDTH, IMAGE_HEIGHT)
    )

    # ------------------------------------------------------
    # RGB Conversion
    # ------------------------------------------------------

    rgb = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2RGB
    )

    # ------------------------------------------------------
    # Gaussian Blur
    # ------------------------------------------------------

    blurred = cv2.GaussianBlur(
        rgb,
        (5, 5),
        0
    )

    # ------------------------------------------------------
    # Grayscale
    # ------------------------------------------------------

    gray = cv2.cvtColor(
        blurred,
        cv2.COLOR_RGB2GRAY
    )

    # ------------------------------------------------------
    # Histogram Equalization
    # ------------------------------------------------------

    equalized = cv2.equalizeHist(gray)

    # ------------------------------------------------------
    # Normalization
    # ------------------------------------------------------

    normalized = equalized / 255.0

    processed = (
        normalized * 255
    ).astype("uint8")

    # ------------------------------------------------------
    # Save Processed Image
    # ------------------------------------------------------

    save_path = os.path.join(
        PROCESSED_FOLDER,
        image_name
    )

    cv2.imwrite(
        save_path,
        processed
    )

    processed_count += 1

    # ------------------------------------------------------
    # Image Statistics
    # ------------------------------------------------------

    height, width = image.shape[:2]

    mean = np.mean(gray)

    std = np.std(gray)

    minimum = np.min(gray)

    maximum = np.max(gray)

    statistics.append(
        [
            image_name,
            width,
            height,
            round(mean, 2),
            round(std, 2),
            minimum,
            maximum
        ]
    )

    # ------------------------------------------------------
    # Save One Comparison Image
    # ------------------------------------------------------

    if not comparison_saved:

        original_gray = cv2.cvtColor(
            resized,
            cv2.COLOR_BGR2GRAY
        )

        comparison = cv2.hconcat(
            [
                original_gray,
                processed
            ]
        )

        cv2.imwrite(

            os.path.join(

                COMPARISON_FOLDER,

                "sample_comparison.png"

            ),

            comparison

        )

        comparison_saved = True

    # ------------------------------------------------------
    # Save One Histogram
    # ------------------------------------------------------

    if not histogram_saved:

        plt.figure(figsize=(7,5))

        plt.hist(
            gray.ravel(),
            bins=256
        )

        plt.title("Pixel Intensity Histogram")

        plt.xlabel("Pixel Value")

        plt.ylabel("Frequency")

        plt.tight_layout()

        plt.savefig(

            os.path.join(

                HISTOGRAM_FOLDER,

                "sample_histogram.png"

            )

        )

        plt.close()

        histogram_saved = True

    # ------------------------------------------------------
    # Save Original & Processed Sample
    # ------------------------------------------------------

    if not sample_saved:

        cv2.imwrite(

            os.path.join(

                SAMPLE_FOLDER,

                "original_sample.png"

            ),

            resized

        )

        cv2.imwrite(

            os.path.join(

                SAMPLE_FOLDER,

                "processed_sample.png"

            ),

            processed

        )

        sample_saved = True

# ==========================================================
# SAVE CSV
# ==========================================================

df = pd.DataFrame(

    statistics,

    columns=[

        "Image Name",

        "Width",

        "Height",

        "Mean Intensity",

        "Standard Deviation",

        "Minimum Pixel",

        "Maximum Pixel"

    ]

)

csv_path = os.path.join(
    OUTPUT_FOLDER,
    "image_statistics.csv"
)

df.to_csv(
    csv_path,
    index=False
)

# ==========================================================
# PREPROCESSING REPORT
# ==========================================================

report_path = os.path.join(
    OUTPUT_FOLDER,
    "preprocessing_report.txt"
)

with open(report_path, "w") as file:

    file.write("=" * 60 + "\n")
    file.write("Bottle Dataset Preprocessing Report\n")
    file.write("=" * 60 + "\n\n")

    file.write("Dataset : MVTec AD Bottle\n\n")

    file.write(f"Training Images : {len(image_files)}\n")
    file.write(f"Successfully Processed : {processed_count}\n\n")

    file.write(f"Image Size : {IMAGE_WIDTH} x {IMAGE_HEIGHT}\n\n")

    file.write("Preprocessing Steps\n")
    file.write("------------------------------\n")
    file.write(" Resize\n")
    file.write(" RGB Conversion\n")
    file.write(" Gaussian Blur\n")
    file.write(" Histogram Equalization\n")
    file.write(" Normalization\n\n")

    file.write("Generated Outputs\n")
    file.write("------------------------------\n")
    file.write(" Processed Images\n")
    file.write(" Image Statistics CSV\n")
    file.write(" Sample Comparison Image\n")
    file.write(" Sample Histogram\n")
    file.write(" Original & Processed Samples\n\n")

    file.write("Status : Completed Successfully\n")

# ==========================================================
# FINAL SUMMARY
# ==========================================================

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETED")
print("=" * 60)

print(f"Images Processed : {processed_count}")
print("Statistics Saved : image_statistics.csv")
print("Comparison Saved : sample_comparison.png")
print("Histogram Saved : sample_histogram.png")
print("Samples Saved : original_sample.png, processed_sample.png")
print("Report Saved : preprocessing_report.txt")

print("=" * 60)