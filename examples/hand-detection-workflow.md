# Hand Detection Training Workflow Example

This example demonstrates a complete ML pipeline for training a 3-class hand detection model (hand/arm/not_hand) using YOLOv8 classification.

## Overview

We built a hand detection system that can distinguish between:
- ✋ **Hand**: Close-up hand with fingers visible
- 💪 **Arm**: Forearm or elbow area without clear hand focus
- ❌ **Not Hand**: Neither hand nor arm

## Complete Workflow

### 1. Data Collection

#### Initial Dataset from Existing Source
```bash
# Copy existing hand detection dataset
cp -r /path/to/ml-visions/hand-detection/hand_cls/* data/hand_cls/

# Structure should be:
data/hand_cls/
├── train/
│   ├── hand/      # Hand images
│   └── not_hand/  # Background images
└── val/
    ├── hand/
    └── not_hand/
```

#### Capture Additional Data
```python
#!/usr/bin/env python3
"""
Capture clean frames for dataset augmentation
Key improvements:
- Clean frames without overlays
- Review before adding to dataset
- Continuous capture without camera reopening
"""

import cv2
import time
from pathlib import Path

class DataCapture:
    def __init__(self):
        self.fps = 10  # 10 frames per second
        self.duration = 20  # 20 seconds per class
        self.temp_dir = Path("temp_captures")

    def capture_class(self, class_name):
        """Capture clean frames without overlays"""
        cap = cv2.VideoCapture(0)
        frames = []

        # Capture at specified FPS
        frame_interval = 1.0 / self.fps
        last_capture = 0
        start_time = time.time()

        while len(frames) < self.fps * self.duration:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = time.time() - start_time

            # Save clean frame at intervals
            if current_time - last_capture >= frame_interval:
                clean_frame = cv2.flip(frame, 1)  # Mirror only
                frames.append(clean_frame)
                last_capture = current_time

            # Show progress to user (not saved)
            display_frame = cv2.flip(frame, 1)
            cv2.putText(display_frame, f"{class_name}: {len(frames)}/{self.fps * self.duration}",
                       (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Capture', display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        return frames
```

### 2. Three-Class Dataset Preparation

```python
#!/usr/bin/env python3
"""
Prepare unified 3-class dataset with proper structure
"""

def prepare_three_class_dataset():
    """Organize data into hand/arm/not_hand structure"""

    data_path = Path("data/hand_cls")

    # Ensure all three classes exist
    for split in ['train', 'val']:
        for cls in ['hand', 'arm', 'not_hand']:
            (data_path / split / cls).mkdir(parents=True, exist_ok=True)

    # Count and balance dataset
    print("📊 Dataset Statistics:")
    for split in ['train', 'val']:
        print(f"\n{split.upper()}:")
        for cls in ['hand', 'arm', 'not_hand']:
            path = data_path / split / cls
            count = len(list(path.glob("*.jpg")))
            print(f"  {cls:10s}: {count:4d} images")

    # 80/20 train/val split for new data
    def split_data(source_dir, train_ratio=0.8):
        images = list(source_dir.glob("*.jpg"))
        train_count = int(len(images) * train_ratio)
        return images[:train_count], images[train_count:]
```

### 3. Training Configuration

```python
#!/usr/bin/env python3
"""
Unified training script with optimized parameters
"""

import subprocess
from pathlib import Path
from datetime import datetime

def train_unified_model():
    """Train 3-class model with YOLOv8"""

    # Model configuration
    model_size = 's'  # 's' for better accuracy than 'n'
    epochs = 50  # Good balance
    batch = 16
    patience = 15  # Early stopping

    # Data path
    data_path = Path("data/hand_cls").absolute()

    # Training timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"unified_{timestamp}"

    print(f"⚙️ Configuration:")
    print(f"   Model: YOLOv8{model_size}-cls")
    print(f"   Epochs: {epochs}")
    print(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Build YOLO command
    cmd = [
        "yolo", "classify", "train",
        f"model=yolov8{model_size}-cls.pt",
        f"data={data_path}",
        f"epochs={epochs}",
        f"batch={batch}",
        f"patience={patience}",
        f"name={run_name}",
        "save=True",
        "exist_ok=True",
        "plots=True",
        "device=mps"  # or "device=0" for CUDA
    ]

    # Run training
    process = subprocess.run(cmd)

    if process.returncode == 0:
        # Save versioned model
        best_model = Path(f"runs/classify/{run_name}/weights/best.pt")
        if best_model.exists():
            models_dir = Path("models")
            models_dir.mkdir(exist_ok=True)

            # Version management
            existing = list(models_dir.glob("unified_v*.pt"))
            version = len(existing) + 1

            import shutil
            dest = models_dir / f"unified_v{version}.pt"
            shutil.copy(best_model, dest)

            # Update latest
            latest = models_dir / "hand_detector.pt"
            shutil.copy(best_model, latest)

            print(f"✅ Model saved: {dest}")
            return dest
```

### 4. Live Demo Implementation

```python
#!/usr/bin/env python3
"""
Live webcam demo with color-coded confidence
"""

import cv2
from ultralytics import YOLO
from pathlib import Path

def run_live_demo(weights='models/hand_detector.pt'):
    """Live hand detection with confidence visualization"""

    if not Path(weights).exists():
        print(f"❌ Model not found: {weights}")
        return

    model = YOLO(weights)
    cap = cv2.VideoCapture(0)

    # Class names
    class_names = ['hand', 'arm', 'not_hand']

    print("🎥 Live Demo Started")
    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Inference
        results = model(frame, verbose=False)

        if results and results[0].probs:
            probs = results[0].probs
            top1_idx = probs.top1
            confidence = probs.top1conf.item()
            class_name = class_names[top1_idx]

            # Color based on confidence
            if confidence > 0.8:
                color = (0, 255, 0)  # Green - High
            elif confidence > 0.5:
                color = (0, 255, 255)  # Yellow - Medium
            else:
                color = (0, 0, 255)  # Red - Low

            # Display results
            cv2.putText(frame, f"{class_name}: {confidence:.1%}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       1, color, 2)

        cv2.imshow('Hand Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
```

## Key Improvements We Made

### 1. Clean Frame Capture
- **Problem**: Overlays were being saved in captured images
- **Solution**: Save clean frames before adding display overlays
```python
clean_frame = cv2.flip(frame, 1)  # Just mirror, no text
cv2.imwrite(filename, clean_frame)

# Then add overlays for display only
display_frame = cv2.flip(frame, 1)
cv2.putText(display_frame, ...)  # Only for display
```

### 2. Continuous Capture
- **Problem**: Camera reopening between captures was jarring
- **Solution**: Keep camera open for entire session
```python
cap = cv2.VideoCapture(0)
# Capture all classes without closing
for class_name in ['hand', 'arm', 'not_hand']:
    capture_class(cap, class_name)
cap.release()
```

### 3. Dataset Structure
- **Problem**: Multiple separate directories for different experiments
- **Solution**: Unified hand_cls structure with all classes
```
data/hand_cls/
├── train/
│   ├── hand/
│   ├── arm/
│   └── not_hand/
└── val/
    ├── hand/
    ├── arm/
    └── not_hand/
```

### 4. Model Versioning
- **Problem**: Models being overwritten
- **Solution**: Automatic versioning with latest pointer
```python
# Save versioned
dest = models_dir / f"unified_v{version}.pt"
# Update latest
latest = models_dir / "hand_detector.pt"
```

### 5. Training Parameters
- **Problem**: Determining optimal epochs and early stopping
- **Solution**: 50 epochs with patience=15
- Loss progression example:
  - Epoch 1: 0.6559
  - Epoch 2: 0.1241 (↓81%)
  - Epoch 3: 0.0506 (↓59%)
  - Epoch 13: 0.0066 (converged)

## Results

- **Training Time**: ~30 seconds per epoch on Apple M1
- **Final Accuracy**: >96% on 3-class detection
- **Dataset Size**: 1740 images (704 hand, 320 arm, 462 not_hand)
- **Model Size**: YOLOv8s-cls (5M parameters)

## Deployment

### Hugging Face Spaces
```python
# app.py for Gradio interface
import gradio as gr
from ultralytics import YOLO

model = YOLO("hand_detector.pt")

def classify_image(image):
    results = model(image)
    probs = results[0].probs
    return {
        "hand": float(probs.data[0]),
        "arm": float(probs.data[1]),
        "not_hand": float(probs.data[2])
    }

iface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(),
    outputs=gr.Label(num_top_classes=3),
    title="Hand Detection Model"
)

iface.launch()
```

## Lessons Learned

1. **Data Quality > Quantity**: 1740 well-labeled images gave 96% accuracy
2. **Clean Captures**: Never save UI overlays in training data
3. **Model Size**: YOLOv8s performed better than YOLOv8n for nuanced detection
4. **Early Stopping**: Patience=15 prevents overfitting while allowing convergence
5. **Version Control**: Always version models, never overwrite
6. **User Experience**: Continuous capture is better than start/stop

## Next Steps

1. Add more diverse data (different lighting, backgrounds)
2. Implement gesture recognition on top of hand detection
3. Export to ONNX/TensorFlow.js for browser deployment
4. Create mobile app with TensorFlow Lite

## Commands Reference

```bash
# Capture new data
python3 capture_arm_only.py

# Train model
python3 train_unified.py

# Test live
python3 live_demo.py

# Deploy to HuggingFace
python3 deploy_huggingface.py
```

## Performance Metrics

| Metric | Value |
|--------|-------|
| Training Loss | 0.0066 |
| Validation Accuracy | 96.3% |
| Inference Speed | 30 FPS |
| Model Size | 20 MB |
| Training Time | 25 minutes |