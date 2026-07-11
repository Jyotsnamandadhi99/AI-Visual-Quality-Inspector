"""
Dataset Analysis - MVTec AD Bottle Dataset

Author : Jyotsna Mandadhi
Project : AI Visual Quality Inspector
"""

import os
import cv2
import random
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# PATHS
# ==========================================================

DATASET_PATH = r"data/raw/bottle"

OUTPUT_PATH = r"model_outputs/dataset_analysis"

os.makedirs(OUTPUT_PATH, exist_ok=True)

# ==========================================================
# DATASET INFORMATION
# ==========================================================

dataset = {
    "Train Good": os.path.join(DATASET_PATH, "train", "good"),
    "Test Good": os.path.join(DATASET_PATH, "test", "good"),
    "Broken Large": os.path.join(DATASET_PATH, "test", "broken_large"),
    "Broken Small": os.path.join(DATASET_PATH, "test", "broken_small"),
    "Contamination": os.path.join(DATASET_PATH, "test", "contamination")
}

# ==========================================================
# COUNT IMAGES
# ==========================================================

statistics = []

print("=" * 60)
print("DATASET ANALYSIS")
print("=" * 60)

for category, folder in dataset.items():

    image_count = len([
        file for file in os.listdir(folder)
        if file.endswith(".png")
    ])

    statistics.append([category, image_count])

    print(f"{category:<20} : {image_count}")

# ==========================================================
# SAVE CSV
# ==========================================================

df = pd.DataFrame(
    statistics,
    columns=["Category", "Images"]
)

df.to_csv(
    os.path.join(
        OUTPUT_PATH,
        "dataset_statistics.csv"
    ),
    index=False
)

# ==========================================================
# BAR CHART
# ==========================================================

plt.figure(figsize=(8,5))

plt.bar(df["Category"], df["Images"])

plt.title("Dataset Distribution")

plt.ylabel("Number of Images")

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_PATH,
        "class_distribution.png"
    )
)

plt.close()

# ==========================================================
# RANDOM IMAGE SAVER
# ==========================================================

def save_random_image(folder, save_name):

    image_name = random.choice(os.listdir(folder))

    image_path = os.path.join(folder, image_name)

    image = cv2.imread(image_path)

    cv2.imwrite(
        os.path.join(
            OUTPUT_PATH,
            save_name
        ),
        image
    )

    return image_name

# ==========================================================
# SAVE RANDOM SAMPLES
# ==========================================================

good_image = save_random_image(
    dataset["Train Good"],
    "random_good.png"
)

broken_large = save_random_image(
    dataset["Broken Large"],
    "random_broken_large.png"
)

broken_small = save_random_image(
    dataset["Broken Small"],
    "random_broken_small.png"
)

contamination = save_random_image(
    dataset["Contamination"],
    "random_contamination.png"
)

# ==========================================================
# SAVE GROUND TRUTH MASK
# ==========================================================

mask_folder = os.path.join(
    DATASET_PATH,
    "ground_truth",
    "broken_large"
)

mask_name = random.choice(os.listdir(mask_folder))

mask = cv2.imread(
    os.path.join(mask_folder, mask_name),
    cv2.IMREAD_GRAYSCALE
)

cv2.imwrite(
    os.path.join(
        OUTPUT_PATH,
        "ground_truth_mask.png"
    ),
    mask
)

# ==========================================================
# IMAGE VS MASK COMPARISON
# ==========================================================

image_name = mask_name.replace("_mask", "")

image_path = os.path.join(
    DATASET_PATH,
    "test",
    "broken_large",
    image_name
)

image = cv2.imread(image_path)

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.title("Bottle")

plt.axis("off")

plt.subplot(1,2,2)

plt.imshow(mask, cmap="gray")

plt.title("Ground Truth")

plt.axis("off")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_PATH,
        "comparison.png"
    )
)

plt.close()

# ==========================================================
# SUMMARY REPORT
# ==========================================================

train_images = 209

test_images = 20 + 20 + 22 + 21

good_images = 209 + 20

defective_images = 20 + 22 + 21

total_images = train_images + test_images

defect_percentage = (
    defective_images /
    total_images
) * 100

summary_path = os.path.join(
    OUTPUT_PATH,
    "dataset_summary.txt"
)

with open(summary_path, "w") as file:

    file.write("MVTec Bottle Dataset Summary\n")

    file.write("="*50 + "\n\n")

    file.write(f"Total Images : {total_images}\n")

    file.write(f"Training Images : {train_images}\n")

    file.write(f"Testing Images : {test_images}\n\n")

    file.write(f"Good Images : {good_images}\n")

    file.write(f"Defective Images : {defective_images}\n\n")

    file.write(f"Defective Percentage : {defect_percentage:.2f}%\n\n")

    file.write("Defect Categories\n")

    file.write("---------------------------\n")

    file.write("Broken Large\n")

    file.write("Broken Small\n")

    file.write("Contamination\n")

print("\nDataset analysis completed successfully!")
print(f"Outputs saved to: {OUTPUT_PATH}")