# AI Visual Quality Inspector

An end-to-end Industrial AI Quality Inspection System that combines **OpenCV**, **Deep Learning**, **Industrial Anomaly Detection**, and **Hugging Face Vision-Language Models** to automatically inspect products, localize defects, generate anomaly heatmaps, and create intelligent quality inspection reports.

---

## Project Overview

This project is being built from scratch to simulate a real-world industrial visual inspection system.

The application learns the appearance of **normal products** using the **MVTec AD Bottle Dataset** and detects manufacturing defects such as:

- Broken Large
- Broken Small
- Contamination

The final system will automatically inspect products and generate an AI-powered inspection report.

---

## Key Features

### OpenCV Pipeline
- Image preprocessing
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
- Scene understanding
- AI-generated inspection summaries

### Industrial Anomaly Detection *(In Progress)*
- MVTec AD Bottle Dataset
- Defect localization
- Anomaly heatmap generation
- PASS / FAIL quality inspection
- AI quality report generation

### Deployment *(Upcoming)*
- Streamlit Web Application
- Interactive Dashboard
- Report Download

---

## Tech Stack

- Python
- OpenCV
- NumPy
- Pandas
- Matplotlib
- PyTorch
- Ultralytics YOLOv8
- Hugging Face Transformers
- Anomalib *(Upcoming)*
- Streamlit *(Upcoming)*

---

## Dataset

### MVTec AD - Bottle Category

Training Images

- Good Bottles : **209**

Testing Images

- Good : **20**
- Broken Large : **20**
- Broken Small : **22**
- Contamination : **21**

Ground Truth

- Pixel-level segmentation masks for all defective images

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
│
│── outputs/
│
│── screenshots/
│
│── src/
│
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
Industrial Anomaly Detection
      │
      ▼
Anomaly Score
      │
      ▼
Defect Heatmap
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

### Phase 1 — Computer Vision Fundamentals ✅

- Image Processing
- Image Transformations
- ROI Operations
- Video Processing
- Webcam Integration

### Phase 2 — Deep Learning ✅

- YOLOv8 Object Detection
- Custom Detection Pipeline
- Object Counting
- Confidence Filtering
- Hugging Face BLIP Captioning
- AI Visual Analysis

### Phase 3 — Industrial AI *(Current Phase)*

- Dataset Exploration
- Dataset Analysis
- Data Preprocessing
- Industrial Anomaly Detection *(Upcoming)*

---

## Roadmap

- ✅ OpenCV Fundamentals
- ✅ YOLOv8 Integration
- ✅ Hugging Face BLIP
- ✅ AI Visual Analysis Pipeline
- 🔄 Industrial Anomaly Detection
- ⏳ Model Training
- ⏳ Heatmap Generation
- ⏳ AI Inspection Engine
- ⏳ Streamlit Dashboard
- ⏳ Final Deployment

---

## Project Status

**Current Stage:** Day 10 / 20

**Status:** 🚧 Under Active Development

---

## Author

**Jyotsna Mandadhi**

Computer Science Engineering Student

AI • Computer Vision • Deep Learning