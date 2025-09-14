---
title: ML Model Inference
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
models:
  - yolov8n
---

# ML Model Inference Platform

Deploy and test your machine learning models on Hugging Face Spaces.

## Features
- Image Classification
- Object Detection (YOLO)
- Text Classification
- Multi-model support

## Configuration

Set these environment variables in your Space settings:
- `MODEL_TYPE`: Type of model (pytorch, yolo, transformers)
- `MODEL_PATH`: Path to model file
- `HF_MODEL_ID`: Hugging Face model ID (optional)