# AI Visual Quality Inspector

An AI-powered visual quality inspection system that combines **OpenCV**, **YOLO**, and **Hugging Face Vision-Language Models** to detect product defects, generate quality control reports, and provide natural language explanations.

## Features

- Image preprocessing using OpenCV
- Product detection with YOLO
- Defect localization and analysis
- Vision-Language AI using Hugging Face
- Automated Pass/Fail quality inspection
- Quality control report generation
- Interactive web interface using Streamlit

## Tech Stack

- Python
- OpenCV
- Hugging Face Transformers
- YOLOv8
- Streamlit
- NumPy
- Pandas
- PyTorch

## Project Structure

```
AI-Visual-Quality-Inspector/
│── data/
│── src/
│── models/
│── outputs/
│── screenshots/
│── requirements.txt
│── README.md
```

## Project Workflow

```
Input Image
      │
      ▼
OpenCV Preprocessing
      │
      ▼
Object Detection (YOLO)
      │
      ▼
Image Segmentation
      │
      ▼
Defect Analysis
      │
      ▼
Hugging Face Vision-Language Model
      │
      ▼
Quality Report Generation
      │
      ▼
Streamlit Web Application
```

## Features

- Image preprocessing (resize, grayscale, RGB conversion, Gaussian blur, edge detection)
- Image transformations (rotation, flipping, translation, scaling, cropping)
- Region of Interest (ROI) extraction and visualization
- Automatic output generation and organized project structure
- Modular OpenCV-based image processing pipeline

## Project Status

🚧 In Progress (Day 3/21)

**Completed**
- OpenCV Fundamentals
- Image Preprocessing
- Image Transformations

**Coming Next**
- Video Processing
- YOLOv8 Object Detection
- Hugging Face Vision Models
- AI Quality Inspection Pipeline
- Streamlit Web Application

## Author

**Jyotsna Mandadhi**