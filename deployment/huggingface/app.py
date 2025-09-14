import gradio as gr
import torch
import numpy as np
from PIL import Image
from transformers import pipeline, AutoModelForImageClassification, AutoTokenizer, AutoModelForSequenceClassification
from ultralytics import YOLO
import json
import os
from typing import Dict, Any, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration
MODEL_TYPE = os.environ.get("MODEL_TYPE", "pytorch")
MODEL_PATH = os.environ.get("MODEL_PATH", "./models/model.pth")
HF_MODEL_ID = os.environ.get("HF_MODEL_ID", None)

# Load model based on type
model = None
processor = None


def load_model():
    """Load the model based on configuration"""
    global model, processor

    try:
        if HF_MODEL_ID:
            # Load from Hugging Face Hub
            if "image-classification" in HF_MODEL_ID:
                model = pipeline("image-classification", model=HF_MODEL_ID)
            elif "text-classification" in HF_MODEL_ID:
                model = pipeline("text-classification", model=HF_MODEL_ID)
            elif "object-detection" in HF_MODEL_ID:
                model = pipeline("object-detection", model=HF_MODEL_ID)
            else:
                model = AutoModelForImageClassification.from_pretrained(HF_MODEL_ID)
            logger.info(f"Model loaded from Hugging Face: {HF_MODEL_ID}")

        elif MODEL_TYPE == "yolo":
            model = YOLO(MODEL_PATH)
            logger.info(f"YOLO model loaded from {MODEL_PATH}")

        elif MODEL_TYPE == "pytorch":
            model = torch.load(MODEL_PATH, map_location="cpu")
            model.eval()
            logger.info(f"PyTorch model loaded from {MODEL_PATH}")

        else:
            raise ValueError(f"Unsupported model type: {MODEL_TYPE}")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def classify_image(image: Image.Image, top_k: int = 5) -> Dict[str, float]:
    """Image classification function"""
    try:
        if isinstance(model, pipeline):
            results = model(image)
            return {r['label']: r['score'] for r in results[:top_k]}

        elif MODEL_TYPE == "yolo":
            results = model(image)
            if results[0].probs:
                top5 = results[0].probs.top5
                top5conf = results[0].probs.top5conf.tolist()
                class_names = model.names if hasattr(model, 'names') else {}
                return {
                    class_names.get(int(cls), f"Class {cls}"): conf
                    for cls, conf in zip(top5, top5conf)
                }

        elif MODEL_TYPE == "pytorch":
            # Custom PyTorch model inference
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            input_tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                top_k_values, top_k_indices = torch.topk(probabilities, top_k)

            return {
                f"Class {idx}": float(val)
                for idx, val in zip(top_k_indices[0], top_k_values[0])
            }

        else:
            return {"error": "Model type not supported for classification"}

    except Exception as e:
        logger.error(f"Classification error: {e}")
        return {"error": str(e)}


def detect_objects(image: Image.Image, conf_threshold: float = 0.25) -> Tuple[Image.Image, List[Dict]]:
    """Object detection function"""
    try:
        detections = []

        if isinstance(model, pipeline) and "object-detection" in str(type(model)):
            results = model(image)
            for r in results:
                if r['score'] >= conf_threshold:
                    detections.append({
                        "label": r['label'],
                        "confidence": r['score'],
                        "box": r['box']
                    })

        elif MODEL_TYPE == "yolo":
            results = model(image, conf=conf_threshold)
            for r in results:
                if hasattr(r, 'boxes'):
                    for box in r.boxes:
                        detections.append({
                            "label": model.names[int(box.cls)] if hasattr(model, 'names') else f"Class {int(box.cls)}",
                            "confidence": float(box.conf),
                            "box": box.xyxy[0].tolist()
                        })

            # Draw boxes on image
            annotated = results[0].plot()
            image = Image.fromarray(annotated)

        return image, detections

    except Exception as e:
        logger.error(f"Detection error: {e}")
        return image, [{"error": str(e)}]


def classify_text(text: str, max_length: int = 512) -> Dict[str, float]:
    """Text classification function"""
    try:
        if isinstance(model, pipeline) and "text-classification" in str(type(model)):
            results = model(text, max_length=max_length, truncation=True)
            return {r['label']: r['score'] for r in results}

        else:
            return {"error": "Model type not supported for text classification"}

    except Exception as e:
        logger.error(f"Text classification error: {e}")
        return {"error": str(e)}


# Create Gradio interface
def create_interface():
    """Create and configure Gradio interface"""

    load_model()

    with gr.Blocks(title="ML Model Inference") as app:
        gr.Markdown("# ML Model Inference Platform")
        gr.Markdown("Deploy and test your machine learning models")

        with gr.Tabs():
            # Image Classification Tab
            with gr.TabItem("Image Classification"):
                with gr.Row():
                    with gr.Column():
                        img_input = gr.Image(type="pil", label="Upload Image")
                        top_k_slider = gr.Slider(1, 10, value=5, step=1, label="Top K Classes")
                        classify_btn = gr.Button("Classify", variant="primary")

                    with gr.Column():
                        class_output = gr.Label(label="Predictions")

                classify_btn.click(
                    fn=classify_image,
                    inputs=[img_input, top_k_slider],
                    outputs=class_output
                )

                gr.Examples(
                    examples=[
                        ["examples/cat.jpg", 5],
                        ["examples/dog.jpg", 5],
                    ],
                    inputs=[img_input, top_k_slider],
                    outputs=class_output,
                    fn=classify_image,
                    cache_examples=True,
                )

            # Object Detection Tab
            if MODEL_TYPE == "yolo" or (HF_MODEL_ID and "object-detection" in HF_MODEL_ID):
                with gr.TabItem("Object Detection"):
                    with gr.Row():
                        with gr.Column():
                            det_input = gr.Image(type="pil", label="Upload Image")
                            conf_slider = gr.Slider(0.1, 0.9, value=0.25, step=0.05, label="Confidence Threshold")
                            detect_btn = gr.Button("Detect Objects", variant="primary")

                        with gr.Column():
                            det_image = gr.Image(label="Annotated Image")
                            det_json = gr.JSON(label="Detections")

                    detect_btn.click(
                        fn=detect_objects,
                        inputs=[det_input, conf_slider],
                        outputs=[det_image, det_json]
                    )

            # Text Classification Tab
            if HF_MODEL_ID and "text-classification" in HF_MODEL_ID:
                with gr.TabItem("Text Classification"):
                    with gr.Row():
                        with gr.Column():
                            text_input = gr.Textbox(lines=5, label="Enter Text")
                            max_len_slider = gr.Slider(32, 1024, value=512, step=32, label="Max Length")
                            text_classify_btn = gr.Button("Classify Text", variant="primary")

                        with gr.Column():
                            text_output = gr.Label(label="Predictions")

                    text_classify_btn.click(
                        fn=classify_text,
                        inputs=[text_input, max_len_slider],
                        outputs=text_output
                    )

            # Model Information Tab
            with gr.TabItem("Model Info"):
                gr.Markdown(f"""
                ## Model Configuration
                - **Model Type**: {MODEL_TYPE}
                - **Model Path**: {MODEL_PATH}
                - **HF Model ID**: {HF_MODEL_ID or "Not specified"}
                - **Device**: {"CUDA" if torch.cuda.is_available() else "CPU"}

                ## API Usage
                You can also use this model via API:
                ```python
                import requests
                response = requests.post(
                    "https://your-space.hf.space/api/predict",
                    json={{"data": [image_base64, 5]}}
                )
                ```
                """)

    return app


# Launch the app
if __name__ == "__main__":
    app = create_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )