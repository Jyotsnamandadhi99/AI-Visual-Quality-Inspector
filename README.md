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
- Ultralytics YOLOv8
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
- Video processing with frame-by-frame analysis
- Real-time webcam integration and image capture
- Custom object detection using YOLOv8 and OpenCV
- Object counting with confidence threshold filtering
- AI-powered image captioning using Hugging Face BLIP
- Integrated visual analysis pipeline combining YOLOv8 and BLIP
- Automatic generation of detection images and AI analysis reports
- Well-structured project organization with documented outputs

## Project Status

In Progress (Day 9/21)

### Completed

#### OpenCV Fundamentals
- Image Preprocessing
- Image Transformations
- ROI Operations
- Video Processing
- Webcam Integration

#### YOLOv8 Integration
- Image Object Detection
- Video Object Detection
- Custom Bounding Box Visualization
- Confidence Threshold Filtering
- Object Counting
- Detection Report Generation

#### Hugging Face Integration
- BLIP Image Captioning
- Caption Report Generation
- Combine YOLO + BLIP into a Single AI Pipeline

### Coming Next
- AI Visual Quality Inspection
- Streamlit Web Application
- Final Project Optimization & Deployment

## Author

**Jyotsna Mandadhi**