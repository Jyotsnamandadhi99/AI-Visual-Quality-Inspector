# AI Visual Quality Inspector

An end-to-end **Industrial AI Visual Quality Inspection System** that combines **OpenCV**, **PyTorch**, **YOLOv8**, **Industrial Anomaly Detection**, and **Hugging Face Vision-Language Models** to automatically inspect products, localize defects, generate anomaly heatmaps, and produce intelligent quality inspection reports.

---

## Project Overview

This project simulates a real-world industrial quality inspection system used in manufacturing.

The system learns the appearance of **normal bottles** using the **MVTec AD Bottle Dataset** and identifies manufacturing defects such as:

- Broken Large
- Broken Small
- Contamination

The final application will automatically classify products as **PASS** or **FAIL**, localize defects, generate AI-powered inspection reports, and provide an interactive Streamlit dashboard.

---

## Key Features

### Computer Vision
- Image preprocessing using OpenCV
- Image enhancement
- ROI extraction
- Video processing
- Webcam integration

### Object Detection
- YOLOv8 object detection
- Custom bounding box visualization
- Confidence threshold filtering
- Object counting

### Vision-Language AI
- Hugging Face BLIP image captioning
- AI-generated image descriptions
- Natural language inspection summaries

### Industrial AI (Current Phase)
- MVTec AD Bottle Dataset
- Custom PyTorch Dataset
- Custom DataLoader Pipeline
- Dataset exploration & preprocessing
- Industrial anomaly detection pipeline *(Upcoming)*
- Defect localization *(Upcoming)*
- Heatmap generation *(Upcoming)*
- PASS / FAIL inspection *(Upcoming)*

### Deployment *(Upcoming)*
- Streamlit Dashboard
- AI Inspection Report Download

---

## Tech Stack

- Python
- OpenCV
- NumPy
- Pandas
- Matplotlib
- PyTorch
- TorchVision
- Ultralytics YOLOv8
- Hugging Face Transformers
- Anomalib
- Streamlit

---

## Dataset

### MVTec AD – Bottle

### Training Set

| Category | Images |
|----------|-------:|
| Good | 209 |

### Testing Set

| Category | Images |
|----------|-------:|
| Good | 20 |
| Broken Large | 20 |
| Broken Small | 22 |
| Contamination | 21 |

### Ground Truth

Pixel-level segmentation masks are provided for all defective images.

---

## Project Structure

```text
AI-Visual-Quality-Inspector/

│── app/
│
│── data/
│   ├── raw/
│   │   └── bottle/
│   └── processed/
│
│── model_src/
│   ├── dataset_analysis.py
│   ├── preprocessing.py
│   ├── dataset_loader.py
│   ├── train.py
│   ├── inference.py
│   └── inspection_engine.py
│
│── model_outputs/
│   ├── dataset_analysis/
│   ├── preprocessing/
│   ├── predictions/
│   ├── reports/
│   └── heatmaps/
│
│── models/
│── outputs/
│── screenshots/
│── src/
│── requirements.txt
│
└── README.md
```

---

## Workflow

```text
Bottle Image
      │
      ▼
OpenCV Preprocessing
      │
      ▼
Custom PyTorch Dataset
      │
      ▼
PyTorch DataLoader
      │
      ▼
Industrial Anomaly Detection
      │
      ▼
Anomaly Score
      │
      ▼
Heatmap Generation
      │
      ▼
PASS / FAIL Decision
      │
      ▼
Hugging Face BLIP
      │
      ▼
AI Inspection Report
      │
      ▼
Streamlit Dashboard
```

---

## Current Progress

### ✅ Phase 1 — Computer Vision

- Image Preprocessing
- Image Transformations
- ROI Operations
- Video Processing
- Webcam Integration

### ✅ Phase 2 — Deep Learning

- YOLOv8 Object Detection
- Custom Detection Pipeline
- Object Counting
- Confidence Threshold Filtering
- Hugging Face BLIP Image Captioning
- AI Visual Analysis Pipeline

### 🚧 Phase 3 — Industrial AI

Completed

- Dataset Exploration
- Dataset Analysis
- Image Preprocessing Pipeline
- Custom PyTorch Dataset
- Custom PyTorch DataLoader

Upcoming

- PatchCore Training
- Heatmap Generation
- Defect Localization
- Inspection Engine
- AI Report Generation

---

## Roadmap

- ✅ OpenCV Fundamentals
- ✅ YOLOv8 Integration
- ✅ Hugging Face BLIP
- ✅ AI Visual Analysis
- ✅ Dataset Analysis
- ✅ Image Preprocessing
- ✅ Custom PyTorch Dataset
- ✅ Custom DataLoader
- 🔄 PatchCore Training
- ⏳ Heatmap Generation
- ⏳ AI Inspection Engine
- ⏳ Streamlit Dashboard
- ⏳ Final Deployment

---

## Project Status

**Current Stage:** Day 12 / 20

**Status:** 🚧 Industrial AI Pipeline Development

---

## Author

**Jyotsna Mandadhi**

Computer Science Engineering Student
