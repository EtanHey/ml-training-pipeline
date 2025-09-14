#!/usr/bin/env python3
"""
Super Simple ML Training - Just run: python train_simple.py

This single file handles everything:
1. Checks your data
2. Trains a model
3. Tests it
4. Saves it with version numbers
"""

import os
import sys
from pathlib import Path

def setup():
    """One-time setup"""
    print("🚀 ML Training - Simple Version\n")

    # Create directories if they don't exist
    Path("data/train").mkdir(parents=True, exist_ok=True)
    Path("data/val").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    # Check if data exists
    train_count = len(list(Path("data/train").rglob("*.*")))
    val_count = len(list(Path("data/val").rglob("*.*")))

    if train_count == 0:
        print("❌ No training data found!\n")
        print("Please add images to data/train/")
        print("Example structure:")
        print("  data/train/cats/cat1.jpg")
        print("  data/train/dogs/dog1.jpg")
        print("\nFolder names = categories")
        sys.exit(1)

    print(f"✅ Found {train_count} training files")
    if val_count > 0:
        print(f"✅ Found {val_count} validation files")
    else:
        print("⚠️  No validation data (will use 20% of training)")

    return train_count, val_count

def get_latest_model():
    """Find the latest model version"""
    models = sorted(Path("models").glob("model_v*.pt"))
    if models:
        return str(models[-1]), len(models) + 1
    return None, 1

def train():
    """Main training function"""
    train_count, val_count = setup()

    # Check dependencies
    try:
        from ultralytics import YOLO
        print("✅ Dependencies OK\n")
    except ImportError:
        print("Installing dependencies (one-time)...")
        os.system("pip install ultralytics --quiet")
        from ultralytics import YOLO

    # Menu
    print("What would you like to do?")
    print("1) Train new model (from scratch)")
    print("2) Continue training (from last model)")
    print("3) Test existing model")
    print("4) Quick test (live webcam)")

    choice = input("\nChoice (1-4): ").strip()

    if choice == "1":
        # Train new model
        print("\n🎯 Starting fresh training...")
        print("This will take 5-10 minutes\n")

        model = YOLO("yolov8n-cls.pt")  # Start with small model

        # Train with simple settings
        results = model.train(
            data="data",
            epochs=10,  # Quick training
            batch=16,
            patience=5,  # Stop if not improving
            save=True,
            device="auto",  # Auto-detect GPU
            verbose=True
        )

        # Save with version number
        _, next_version = get_latest_model()
        save_path = f"models/model_v{next_version}.pt"
        model.save(save_path)

        print(f"\n✅ Model saved as {save_path}")
        print(f"   Accuracy: {results.metrics.get('accuracy', 'N/A')}")

        # Log the training
        with open("model_history.log", "a") as f:
            f.write(f"v{next_version}: {train_count} images, 10 epochs\n")

    elif choice == "2":
        # Continue training
        latest_model, next_version = get_latest_model()
        if not latest_model:
            print("❌ No existing model found! Train a new one first.")
            return

        print(f"\n🔄 Continuing from {latest_model}...")

        model = YOLO(latest_model)
        results = model.train(
            data="data",
            epochs=10,
            batch=16,
            resume=True
        )

        # Save new version
        save_path = f"models/model_v{next_version}.pt"
        model.save(save_path)

        print(f"\n✅ New version saved as {save_path}")

    elif choice == "3":
        # Test model
        latest_model, _ = get_latest_model()
        if not latest_model:
            print("❌ No model found! Train one first.")
            return

        print(f"\n🧪 Testing {latest_model}...")
        print("Starting web interface at http://localhost:7860\n")

        try:
            import gradio as gr
        except ImportError:
            print("Installing test interface...")
            os.system("pip install gradio --quiet")
            import gradio as gr

        model = YOLO(latest_model)

        def predict(img):
            results = model(img)
            return results[0].plot() if hasattr(results[0], 'plot') else img

        gr.Interface(
            fn=predict,
            inputs=gr.Image(),
            outputs=gr.Image(),
            title=f"Testing: {latest_model}",
            description="Upload an image to test your model"
        ).launch()

    elif choice == "4":
        # Quick webcam test
        latest_model, _ = get_latest_model()
        if not latest_model:
            print("❌ No model found! Train one first.")
            return

        print(f"\n📷 Starting webcam test with {latest_model}")
        print("Press 'q' to quit\n")

        model = YOLO(latest_model)

        # Run on webcam
        model.predict(source=0, show=True)

    else:
        print("Invalid choice")

def main():
    """Entry point"""
    print("=" * 50)
    print("     SIMPLE ML TRAINING SCRIPT")
    print("=" * 50)

    # Check Python version
    if sys.version_info < (3, 7):
        print("❌ Python 3.7+ required")
        sys.exit(1)

    try:
        train()
    except KeyboardInterrupt:
        print("\n\n👋 Stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nCommon fixes:")
        print("1. Make sure you have images in data/train/")
        print("2. Check folder structure (folders = categories)")
        print("3. Try with fewer images first")

if __name__ == "__main__":
    main()