"""
Bottle Dataset Loader

Day 12

Author : Jyotsna Mandadhi
Project : AI Visual Quality Inspector
"""

import os
from PIL import Image

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision import transforms

# ==========================================================
# DATASET PATH
# ==========================================================

DATASET_PATH = r"data/raw/bottle"

TRAIN_PATH = os.path.join(DATASET_PATH, "train", "good")

TEST_GOOD_PATH = os.path.join(DATASET_PATH, "test", "good")

TEST_BROKEN_LARGE = os.path.join(DATASET_PATH, "test", "broken_large")

TEST_BROKEN_SMALL = os.path.join(DATASET_PATH, "test", "broken_small")

TEST_CONTAMINATION = os.path.join(DATASET_PATH, "test", "contamination")

# ==========================================================
# IMAGE TRANSFORM
# ==========================================================

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# ==========================================================
# CUSTOM DATASET
# ==========================================================

class BottleDataset(Dataset):

    def __init__(self, folder_path, transform=None):

        self.folder_path = folder_path

        self.transform = transform

        self.images = sorted([
            file for file in os.listdir(folder_path)
            if file.endswith(".png")
        ])

    def __len__(self):

        return len(self.images)

    def __getitem__(self, index):

        image_name = self.images[index]

        image_path = os.path.join(
            self.folder_path,
            image_name
        )

        image = Image.open(image_path).convert("RGB")

        if self.transform:

            image = self.transform(image)

        return image, image_name

# ==========================================================
# CREATE DATASETS
# ==========================================================

train_dataset = BottleDataset(
    TRAIN_PATH,
    transform
)

test_good_dataset = BottleDataset(
    TEST_GOOD_PATH,
    transform
)

broken_large_dataset = BottleDataset(
    TEST_BROKEN_LARGE,
    transform
)

broken_small_dataset = BottleDataset(
    TEST_BROKEN_SMALL,
    transform
)

contamination_dataset = BottleDataset(
    TEST_CONTAMINATION,
    transform
)

# ==========================================================
# CREATE DATALOADERS
# ==========================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True
)

test_loader = DataLoader(
    test_good_dataset,
    batch_size=8,
    shuffle=False
)

# ==========================================================
# DATASET SUMMARY
# ==========================================================

print("="*60)

print("Bottle Dataset Loaded")

print("="*60)

print()

print("Training Images :", len(train_dataset))

print("Testing Good :", len(test_good_dataset))

print("Broken Large :", len(broken_large_dataset))

print("Broken Small :", len(broken_small_dataset))

print("Contamination :", len(contamination_dataset))

print()

print("Training Batches :", len(train_loader))

print("Testing Batches :", len(test_loader))

print()

print("="*60)

# ==========================================================
# DISPLAY ONE BATCH
# ==========================================================

images, image_names = next(iter(train_loader))

print()

print("Batch Shape")

print(images.shape)

print()

print("Image Names")

for name in image_names:

    print(name)

print()

print("="*60)

# ==========================================================
# DISPLAY SAMPLE INFORMATION
# ==========================================================

sample_image, sample_name = train_dataset[0]

print()

print("First Image")

print(sample_name)

print()

print("Tensor Shape")

print(sample_image.shape)

print()

print("Minimum Pixel :", sample_image.min().item())

print("Maximum Pixel :", sample_image.max().item())

print()

print("="*60)