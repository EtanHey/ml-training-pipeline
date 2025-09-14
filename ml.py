#!/usr/bin/env python3
"""
Minimal ML Pipeline - Everything in one file for easy copying
Just copy this file to your project and run!
"""

import argparse
import os
import sys
import yaml
import json
from pathlib import Path
from datetime import datetime
import subprocess
import time

# Only import what's available
try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import gradio as gr
    HAS_GRADIO = True
except ImportError:
    HAS_GRADIO = False


class MLPipeline:
    """Single class that handles everything"""

    def __init__(self, project_name="ml_project"):
        self.project_name = project_name
        self.setup_dirs()
        self.load_or_create_config()

    def setup_dirs(self):
        """Create minimal directory structure"""
        Path("datasets/train").mkdir(parents=True, exist_ok=True)
        Path("datasets/val").mkdir(parents=True, exist_ok=True)
        Path("models").mkdir(exist_ok=True)
        Path("logs").mkdir(exist_ok=True)

    def load_or_create_config(self):
        """Load or create default config"""
        config_path = Path("config.yaml")
        if config_path.exists():
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {
                'model_type': 'yolo',
                'epochs': 10,
                'batch_size': 16,
                'learning_rate': 0.001,
                'device': 'auto'
            }
            with open(config_path, 'w') as f:
                yaml.dump(self.config, f)
            print(f"Created default config.yaml")

    def train(self, epochs=None, resume=False):
        """Train model with automatic detection of framework"""
        epochs = epochs or self.config.get('epochs', 10)

        # Auto-detect model type based on what's installed
        if HAS_YOLO and self.config.get('model_type') == 'yolo':
            return self.train_yolo(epochs, resume)
        elif HAS_TORCH:
            return self.train_pytorch(epochs, resume)
        else:
            print("No training framework available. Install ultralytics or torch.")
            return None

    def train_yolo(self, epochs, resume):
        """YOLO training"""
        print(f"Training YOLO for {epochs} epochs...")

        # Find latest model or use base
        models = sorted(Path("models").glob("*.pt"))
        base_model = str(models[-1]) if models and resume else "yolov8n.pt"

        model = YOLO(base_model)
        results = model.train(
            data="datasets",
            epochs=epochs,
            batch=self.config.get('batch_size', 16),
            device=0 if torch.cuda.is_available() else 'cpu'
        )

        # Save with version number
        version = len(list(Path("models").glob("*.pt"))) + 1
        save_path = f"models/model_v{version}.pt"
        model.save(save_path)
        print(f"Model saved to {save_path}")
        return save_path

    def train_pytorch(self, epochs, resume):
        """Generic PyTorch training"""
        print(f"Training PyTorch model for {epochs} epochs...")
        # Add your PyTorch training logic here
        save_path = f"models/model_{datetime.now():%Y%m%d_%H%M%S}.pt"
        # torch.save(model.state_dict(), save_path)
        return save_path

    def test(self, model_path=None):
        """Test model with Gradio if available, else command line"""
        if not model_path:
            models = sorted(Path("models").glob("*.pt"))
            if not models:
                print("No models found in models/")
                return
            model_path = str(models[-1])

        print(f"Testing model: {model_path}")

        if HAS_GRADIO:
            self.test_gradio(model_path)
        else:
            self.test_cli(model_path)

    def test_gradio(self, model_path):
        """Launch Gradio interface"""
        if HAS_YOLO:
            model = YOLO(model_path)

            def predict(img):
                results = model(img)
                return results[0].plot() if hasattr(results[0], 'plot') else img

            gr.Interface(
                fn=predict,
                inputs="image",
                outputs="image",
                title=f"Testing: {model_path}"
            ).launch()
        else:
            print("Gradio interface requires YOLO")

    def test_cli(self, model_path):
        """Command line testing"""
        test_dir = Path("datasets/val")
        if not test_dir.exists():
            test_dir = Path("test_samples")
            test_dir.mkdir(exist_ok=True)

        print(f"Place test images in {test_dir}/")
        print(f"Model loaded: {model_path}")

        if HAS_YOLO:
            model = YOLO(model_path)
            for img_path in test_dir.glob("*.jpg"):
                print(f"Testing {img_path}...")
                results = model(str(img_path))
                print(f"Results: {results}")

    def deploy(self, target="local"):
        """Deploy model to selected target"""
        models = sorted(Path("models").glob("*.pt"))
        if not models:
            print("No models to deploy")
            return

        model_path = models[-1]
        print(f"Deploying {model_path} to {target}...")

        if target == "huggingface":
            self.deploy_huggingface(model_path)
        elif target == "runpod":
            self.deploy_runpod(model_path)
        else:
            print(f"Starting local server with {model_path}...")
            self.test_gradio(str(model_path))

    def deploy_huggingface(self, model_path):
        """Deploy to Hugging Face"""
        print("Deploying to Hugging Face...")
        # Simplified HF deployment
        cmd = f"""
        pip install huggingface-hub
        huggingface-cli login
        huggingface-cli upload {model_path}
        """
        print(f"Run these commands:\n{cmd}")

    def deploy_runpod(self, model_path):
        """Deploy to RunPod"""
        print("Deploying to RunPod...")
        # Create simple handler
        handler_code = f"""
import runpod
from ultralytics import YOLO

model = YOLO("/models/model.pt")

def handler(job):
    image = job["input"]["image"]
    results = model(image)
    return {{"output": results}}

runpod.serverless.start({{"handler": handler}})
"""
        with open("handler.py", "w") as f:
            f.write(handler_code)
        print("Created handler.py - now deploy to RunPod")

    def monitor(self):
        """Monitor training progress"""
        log_files = sorted(Path("logs").glob("*.log"))
        if not log_files:
            print("No training logs found")
            return

        latest_log = log_files[-1]
        print(f"Monitoring {latest_log}")

        # Tail the log file
        subprocess.run(["tail", "-f", str(latest_log)])

    def compare(self, model1=None, model2=None):
        """Compare two models"""
        models = sorted(Path("models").glob("*.pt"))
        if len(models) < 2:
            print("Need at least 2 models to compare")
            return

        model1 = model1 or str(models[-2])
        model2 = model2 or str(models[-1])

        print(f"Comparing:\n  Old: {model1}\n  New: {model2}")

        if HAS_YOLO:
            # Run inference on test set with both models
            m1 = YOLO(model1)
            m2 = YOLO(model2)

            # Add comparison logic here
            print("Run inference on test set to compare...")


def main():
    parser = argparse.ArgumentParser(description="Minimal ML Pipeline")
    parser.add_argument('command', choices=['train', 'test', 'deploy', 'monitor', 'compare', 'setup'],
                        help='Command to run')
    parser.add_argument('--epochs', type=int, help='Training epochs')
    parser.add_argument('--model', type=str, help='Model path')
    parser.add_argument('--target', type=str, default='local',
                        choices=['local', 'huggingface', 'runpod'],
                        help='Deployment target')
    parser.add_argument('--resume', action='store_true', help='Resume from latest checkpoint')

    args = parser.parse_args()

    # Initialize pipeline
    ml = MLPipeline()

    # Execute command
    if args.command == 'train':
        ml.train(args.epochs, args.resume)
    elif args.command == 'test':
        ml.test(args.model)
    elif args.command == 'deploy':
        ml.deploy(args.target)
    elif args.command == 'monitor':
        ml.monitor()
    elif args.command == 'compare':
        ml.compare()
    elif args.command == 'setup':
        print("✅ Project setup complete!")
        print("\nNext steps:")
        print("1. Add training data to datasets/train and datasets/val")
        print("2. Run: python ml.py train")
        print("3. Test: python ml.py test")
        print("4. Deploy: python ml.py deploy --target huggingface")


if __name__ == "__main__":
    main()