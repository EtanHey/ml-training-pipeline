#!/bin/bash
# Modular ML Pipeline Setup - Choose only what you need

echo "🚀 ML Pipeline Setup - Modular Configuration"
echo "============================================"
echo ""
echo "Select components for this project:"
echo ""

# Component selection
COMPONENTS=()

echo "📊 Training Frameworks:"
echo "  1) YOLO (object detection/classification)"
echo "  2) Transformers (NLP/text)"
echo "  3) Custom PyTorch (flexible)"
echo "  4) None (inference only)"
read -p "Select (1-4): " FRAMEWORK_CHOICE

echo ""
echo "🚢 Deployment Targets:"
echo "  1) Hugging Face (web demo, no Docker)"
echo "  2) RunPod (GPU API, Docker)"
echo "  3) Browser (client-side JS)"
echo "  4) Local only (no deployment)"
read -p "Select (1-4 or multiple like 1,2): " DEPLOY_CHOICE

echo ""
echo "📈 Experiment Tracking:"
echo "  1) Weights & Biases"
echo "  2) MLflow"
echo "  3) TensorBoard only"
echo "  4) None (basic logging)"
read -p "Select (1-4): " TRACKING_CHOICE

echo ""
echo "🔧 Additional Tools:"
read -p "GitHub Actions CI/CD? (y/n): " USE_CI
read -p "Background training scripts? (y/n): " USE_BACKGROUND
read -p "Interactive testing UI? (y/n): " USE_TESTING
read -p "Data capture tools? (y/n): " USE_CAPTURE

# Create minimal project structure
echo ""
echo "Creating minimal project structure..."

# Always needed
mkdir -p logs datasets experiments/configs

# Base training script (simplified)
cat > train.py << 'EOF'
#!/usr/bin/env python3
"""
Minimal training script - just the essentials
"""
import argparse
import yaml
from pathlib import Path

def train(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_type = config.get('model_type', 'yolo')

    if model_type == 'yolo':
        from ultralytics import YOLO
        model = YOLO(config.get('model', 'yolov8n.pt'))
        model.train(
            data=config.get('data', 'datasets/'),
            epochs=config.get('epochs', 10),
            batch=config.get('batch', 16)
        )
    else:
        print(f"Training {model_type}...")
        # Add your training logic here

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    args = parser.parse_args()
    train(args.config)
EOF
chmod +x train.py

# Create minimal config
cat > config.yaml << EOF
model_type: yolo
model: yolov8n.pt
data: datasets/
epochs: 10
batch: 16
device: auto
EOF

# Create requirements based on selection
echo "# Core requirements" > requirements.txt
echo "pyyaml>=6.0" >> requirements.txt
echo "numpy>=1.24.0" >> requirements.txt

case $FRAMEWORK_CHOICE in
    1)
        echo "ultralytics>=8.0.0" >> requirements.txt
        echo "torch>=2.0.0" >> requirements.txt
        cp training/yolo_trainer.py ./ 2>/dev/null || true
        ;;
    2)
        echo "transformers>=4.35.0" >> requirements.txt
        echo "torch>=2.0.0" >> requirements.txt
        cp training/transformers_trainer.py ./ 2>/dev/null || true
        ;;
    3)
        echo "torch>=2.0.0" >> requirements.txt
        echo "torchvision>=0.15.0" >> requirements.txt
        cp training/custom_trainer.py ./ 2>/dev/null || true
        ;;
esac

# Deployment setup
IFS=',' read -ra DEPLOY_ARRAY <<< "$DEPLOY_CHOICE"
for choice in "${DEPLOY_ARRAY[@]}"; do
    case $choice in
        1)
            echo "gradio>=4.0.0" >> requirements.txt
            mkdir -p deploy
            cat > deploy/app.py << 'EOF'
import gradio as gr
from ultralytics import YOLO

model = YOLO("best.pt")

def predict(img):
    results = model(img)
    return results[0].plot()

gr.Interface(fn=predict, inputs="image", outputs="image").launch()
EOF
            ;;
        2)
            echo "runpod>=1.6.0" >> requirements.txt
            mkdir -p deploy
            cp deployment/runpod/handler.py deploy/ 2>/dev/null || true
            ;;
        3)
            mkdir -p deploy
            echo '{"name": "ml-inference", "version": "1.0.0"}' > deploy/package.json
            ;;
    esac
done

# Tracking setup
case $TRACKING_CHOICE in
    1)
        echo "wandb>=0.15.0" >> requirements.txt
        ;;
    2)
        echo "mlflow>=2.7.0" >> requirements.txt
        ;;
    3)
        echo "tensorboard>=2.13.0" >> requirements.txt
        ;;
esac

# Optional tools
if [ "$USE_BACKGROUND" = "y" ]; then
    cat > run.sh << 'EOF'
#!/bin/bash
# Simple background training
python train.py > logs/training_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo $! > .pid
echo "Training started (PID: $(cat .pid))"
echo "Monitor: tail -f logs/*.log"
EOF
    chmod +x run.sh
fi

if [ "$USE_TESTING" = "y" ]; then
    cat > test.py << 'EOF'
#!/usr/bin/env python3
import gradio as gr
import sys

model_path = sys.argv[1] if len(sys.argv) > 1 else "best.pt"

# Your test interface here
print(f"Testing {model_path} at http://localhost:7860")
EOF
    chmod +x test.py
fi

if [ "$USE_CAPTURE" = "y" ]; then
    cp /Users/etanheyman/Desktop/Gits/ml-visions/finetune-workshop/capture_and_prepare.py ./ 2>/dev/null || true
fi

# Clean script to remove unused components
cat > clean.sh << 'EOF'
#!/bin/bash
# Remove unused components
echo "Removing unused files..."

# Remove training frameworks not in use
[ ! -f "yolo_trainer.py" ] && rm -rf training/yolo* 2>/dev/null
[ ! -f "transformers_trainer.py" ] && rm -rf training/trans* 2>/dev/null
[ ! -f "custom_trainer.py" ] && rm -rf training/custom* 2>/dev/null

# Remove deployment folders not in use
[ ! -d "deploy" ] && rm -rf deployment/ 2>/dev/null

# Remove CI if not used
[ ! -d ".github" ] && rm -rf .github 2>/dev/null

# Remove heavy docs
rm -rf docs/ 2>/dev/null

echo "✅ Cleaned up unused components"
EOF
chmod +x clean.sh

# Create minimal README
cat > README.md << EOF
# ML Project

## Quick Start

\`\`\`bash
# Install dependencies
pip install -r requirements.txt

# Train
python train.py --config config.yaml

# Test (if enabled)
python test.py best.pt

# Deploy (if configured)
cd deploy && python app.py
\`\`\`

## Project Structure
- \`train.py\` - Main training script
- \`config.yaml\` - Training configuration
- \`datasets/\` - Your data goes here
- \`logs/\` - Training logs
- \`experiments/\` - Saved models and metrics

## Selected Components
- Framework: $FRAMEWORK_CHOICE
- Deployment: $DEPLOY_CHOICE
- Tracking: $TRACKING_CHOICE
EOF

echo ""
echo "✅ Setup complete! Minimal project created with:"
echo ""
echo "📦 Included:"
ls -la *.py *.yaml *.sh 2>/dev/null | awk '{print "  - " $9}'
echo ""
echo "🗑️  To remove unused components: ./clean.sh"
echo "🚀 To start training: python train.py"
echo ""
echo "Total size: $(du -sh . | cut -f1)"