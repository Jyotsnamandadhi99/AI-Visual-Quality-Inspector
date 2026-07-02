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

## Current Features

- Read images using OpenCV
- Resize images to a standard resolution
- Convert images to Grayscale and RGB
- Apply Gaussian Blur for noise reduction
- Perform Canny Edge Detection
- Save processed images automatically

## Project Status

🚧 In Progress (Day 2/37)

### Completed
- ✅ Project setup
- ✅ Virtual environment configuration
- ✅ OpenCV installation
- ✅ Image loading and validation
- ✅ Image resizing
- ✅ Grayscale conversion
- ✅ RGB color space conversion
- ✅ Gaussian Blur
- ✅ Canny Edge Detection
- ✅ Processed image export

## Author

**Jyotsna Mandadhi**