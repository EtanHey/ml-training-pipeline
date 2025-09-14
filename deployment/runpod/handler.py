import runpod
import torch
import numpy as np
from typing import Dict, Any, List, Optional
import base64
import io
from PIL import Image
import json
import logging
import os
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "/models/model.pth")
MODEL_TYPE = os.environ.get("MODEL_TYPE", "pytorch")  # pytorch, onnx, transformers
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = None
tokenizer = None


def load_model():
    """Load model on container startup"""
    global model, tokenizer

    try:
        if MODEL_TYPE == "pytorch":
            model = torch.load(MODEL_PATH, map_location=DEVICE)
            model.eval()
            logger.info(f"PyTorch model loaded from {MODEL_PATH}")

        elif MODEL_TYPE == "onnx":
            import onnxruntime as ort
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if DEVICE == "cuda" else ['CPUExecutionProvider']
            model = ort.InferenceSession(MODEL_PATH, providers=providers)
            logger.info(f"ONNX model loaded from {MODEL_PATH}")

        elif MODEL_TYPE == "transformers":
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            model_name = os.environ.get("MODEL_NAME", "bert-base-uncased")
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH or model_name)
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH or model_name)
            model.to(DEVICE)
            model.eval()
            logger.info(f"Transformers model loaded from {MODEL_PATH or model_name}")

        elif MODEL_TYPE == "yolo":
            from ultralytics import YOLO
            model = YOLO(MODEL_PATH)
            logger.info(f"YOLO model loaded from {MODEL_PATH}")

        else:
            raise ValueError(f"Unsupported model type: {MODEL_TYPE}")

    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise


def decode_image(image_data: str) -> Image.Image:
    """Decode base64 image string to PIL Image"""
    if image_data.startswith('data:image'):
        image_data = image_data.split(',')[1]

    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes))

    if image.mode != 'RGB':
        image = image.convert('RGB')

    return image


def preprocess_image(image: Image.Image, size: tuple = (224, 224)) -> torch.Tensor:
    """Preprocess image for model input"""
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return transform(image).unsqueeze(0).to(DEVICE)


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod handler function

    Expected input format:
    {
        "input": {
            "task": "classify|detect|generate",
            "image": "base64_encoded_image_string",
            "text": "input text for NLP models",
            "parameters": {
                "threshold": 0.5,
                "max_length": 512,
                ...
            }
        }
    }
    """
    try:
        job_input = job.get("input", {})
        task = job_input.get("task", "classify")
        parameters = job_input.get("parameters", {})

        if model is None:
            load_model()

        if task == "classify":
            if "image" in job_input:
                # Image classification
                image = decode_image(job_input["image"])

                if MODEL_TYPE == "yolo":
                    results = model(image, **parameters)
                    predictions = []
                    for r in results:
                        if hasattr(r, 'probs'):  # Classification
                            top5 = r.probs.top5
                            top5conf = r.probs.top5conf.tolist()
                            predictions.append([
                                {"class": int(cls), "confidence": float(conf)}
                                for cls, conf in zip(top5, top5conf)
                            ])
                    return {"predictions": predictions}

                else:
                    input_tensor = preprocess_image(image, size=parameters.get("image_size", (224, 224)))

                    with torch.no_grad():
                        outputs = model(input_tensor)
                        probabilities = torch.nn.functional.softmax(outputs, dim=1)
                        top_k = parameters.get("top_k", 5)
                        values, indices = torch.topk(probabilities, top_k)

                    predictions = [
                        {"class": int(idx), "confidence": float(val)}
                        for idx, val in zip(indices[0], values[0])
                    ]

                    return {"predictions": predictions}

            elif "text" in job_input:
                # Text classification
                if MODEL_TYPE != "transformers":
                    return {"error": "Text classification requires transformers model"}

                text = job_input["text"]
                max_length = parameters.get("max_length", 512)

                inputs = tokenizer(text, return_tensors="pt", truncation=True,
                                 max_length=max_length, padding=True)
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs)
                    probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
                    predicted_class = torch.argmax(probabilities, dim=1)

                return {
                    "predicted_class": int(predicted_class[0]),
                    "confidence": float(probabilities[0][predicted_class[0]]),
                    "all_probabilities": probabilities[0].tolist()
                }

        elif task == "detect":
            if MODEL_TYPE != "yolo":
                return {"error": "Object detection requires YOLO model"}

            image = decode_image(job_input["image"])
            conf_threshold = parameters.get("confidence", 0.25)

            results = model(image, conf=conf_threshold, **parameters)
            detections = []

            for r in results:
                if hasattr(r, 'boxes'):
                    for box in r.boxes:
                        detection = {
                            "bbox": box.xyxy[0].tolist(),  # [x1, y1, x2, y2]
                            "confidence": float(box.conf),
                            "class": int(box.cls),
                            "class_name": model.names[int(box.cls)] if hasattr(model, 'names') else str(int(box.cls))
                        }
                        detections.append(detection)

            return {"detections": detections}

        elif task == "generate":
            # For generative models
            if MODEL_TYPE != "transformers":
                return {"error": "Generation requires transformers model"}

            prompt = job_input.get("prompt", "")
            max_length = parameters.get("max_length", 100)
            temperature = parameters.get("temperature", 1.0)

            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    **parameters
                )

            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            return {"generated_text": generated_text}

        else:
            return {"error": f"Unsupported task: {task}"}

    except Exception as e:
        logger.error(f"Error in handler: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }


# Initialize model on startup
if os.environ.get("LOAD_ON_START", "true").lower() == "true":
    try:
        load_model()
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")

# RunPod serverless handler
runpod.serverless.start({"handler": handler})