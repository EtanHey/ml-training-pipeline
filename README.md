# ML Training Pipeline

A flexible machine learning training pipeline that supports multiple models and deployment options.

## Features

- 🔄 **Model Agnostic**: Easy switching between YOLO, TensorFlow, PyTorch models
- 🚀 **Multiple Deployment Options**: Support for Hugging Face and RunPod
- 📦 **TypeScript Integration**: Browser-based inference with TensorFlow.js/Transformers.js
- 🐍 **Python Training**: Robust training pipeline with popular ML frameworks
- 🔧 **Automation**: CI/CD pipelines for training and deployment
- 📊 **Version Control**: Track models, datasets, and experiments

## Project Structure

```
ml-training-pipeline/
├── training/               # Python training code
│   ├── models/            # Model implementations
│   ├── datasets/          # Dataset handling
│   └── configs/           # Training configurations
├── inference/             # TypeScript/JS inference
│   ├── browser/          # Browser-based inference
│   └── server/           # Node.js server
├── deployment/            # Deployment configurations
│   ├── huggingface/      # HF deployment configs
│   └── runpod/           # RunPod deployment configs
├── automation/            # Automation scripts
└── experiments/           # Experiment tracking
```

## Quick Start

### Training
```bash
# Install dependencies
pip install -r requirements.txt

# Train with default model (YOLO)
python train.py --config configs/default.yaml

# Train with different model
python train.py --model tensorflow --config configs/tensorflow.yaml
```

### Deployment

#### Hugging Face
```bash
./deploy.sh huggingface
```

#### RunPod
```bash
./deploy.sh runpod
```

## Supported Models

- YOLOv8 (Computer Vision)
- TensorFlow/Keras models
- PyTorch models
- Hugging Face Transformers
- Custom models (bring your own)

## Cost Comparison

| Service | GPU | Price/hr | Billing | Best For |
|---------|-----|----------|---------|----------|
| RunPod Serverless | Various | From $0.34 | Per millisecond | Sporadic inference |
| RunPod Pods | RTX 4090 | $0.34 | Per hour | Training |
| HF Inference | Various | From $0.50 | Per minute | Quick deployment |
| HF Spaces | T4 | Free/PRO | Monthly | Demos |

## License

MIT