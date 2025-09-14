FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy handler
COPY handler.py /workspace/handler.py

# Create model directory
RUN mkdir -p /models

# Copy model if provided at build time (optional)
# COPY model.pth /models/model.pth

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/models/model.pth
ENV MODEL_TYPE=pytorch
ENV LOAD_ON_START=true

# RunPod expects the handler at this specific location
CMD ["python", "-u", "handler.py"]