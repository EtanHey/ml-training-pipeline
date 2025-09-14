from ultralytics import YOLO
import torch
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import yaml
import shutil
from base_trainer import BaseTrainer
import logging

logger = logging.getLogger(__name__)


class YOLOTrainer(BaseTrainer):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = config.get("model_name", "yolov8n")
        self.task = config.get("task", "detect")  # detect, segment, classify, pose

    def load_data(self) -> Tuple[Any, Any]:
        data_yaml_path = Path(self.config["data_yaml"])
        if not data_yaml_path.exists():
            self._create_data_yaml()
            data_yaml_path = Path(self.config["data_yaml"])

        with open(data_yaml_path, 'r') as f:
            self.data_config = yaml.safe_load(f)

        logger.info(f"Data configuration loaded from {data_yaml_path}")
        return data_yaml_path, None  # YOLO handles data loading internally

    def _create_data_yaml(self):
        data_config = {
            "path": self.config.get("dataset_path", "./datasets"),
            "train": "train/images",
            "val": "val/images",
            "test": "test/images" if Path(self.config.get("dataset_path", "./datasets"), "test").exists() else None,
            "nc": self.config.get("num_classes", 2),
            "names": self.config.get("class_names", ["class0", "class1"])
        }

        yaml_path = Path(self.experiment_dir / "data.yaml")
        with open(yaml_path, 'w') as f:
            yaml.dump(data_config, f)

        self.config["data_yaml"] = str(yaml_path)
        logger.info(f"Created data.yaml at {yaml_path}")

    def build_model(self) -> YOLO:
        if self.config.get("pretrained_path"):
            model = YOLO(self.config["pretrained_path"])
            logger.info(f"Loaded pretrained model from {self.config['pretrained_path']}")
        else:
            model = YOLO(f"{self.model_name}.pt")
            logger.info(f"Loaded base model: {self.model_name}")

        return model

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        pass

    def validate(self) -> Dict[str, float]:
        pass

    def train(self):
        logger.info("Starting YOLO training...")
        data_yaml, _ = self.load_data()
        self.model = self.build_model()

        training_args = {
            "data": str(data_yaml),
            "epochs": self.config.get("epochs", 100),
            "imgsz": self.config.get("image_size", 640),
            "batch": self.config.get("batch_size", 16),
            "device": 0 if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
            "project": str(self.experiment_dir),
            "name": "train",
            "exist_ok": True,
            "pretrained": self.config.get("pretrained", True),
            "optimizer": self.config.get("optimizer", "SGD"),
            "lr0": self.config.get("learning_rate", 0.01),
            "lrf": self.config.get("lr_final", 0.01),
            "momentum": self.config.get("momentum", 0.937),
            "weight_decay": self.config.get("weight_decay", 0.0005),
            "warmup_epochs": self.config.get("warmup_epochs", 3.0),
            "warmup_momentum": self.config.get("warmup_momentum", 0.8),
            "warmup_bias_lr": self.config.get("warmup_bias_lr", 0.1),
            "amp": self.config.get("amp", True),
            "patience": self.config.get("early_stopping_patience", 50),
            "save": True,
            "save_period": self.config.get("save_interval", 10),
            "val": True,
            "plots": True,
            "resume": self.config.get("resume", False),
        }

        results = self.model.train(**training_args)

        best_model_path = Path(self.experiment_dir) / "train" / "weights" / "best.pt"
        if best_model_path.exists():
            shutil.copy(best_model_path, self.experiment_dir / "best_model.pt")
            logger.info(f"Best model saved to {self.experiment_dir / 'best_model.pt'}")

        return results

    def export_model(self, format: str = "onnx", **kwargs):
        if self.model is None:
            self.model = YOLO(self.experiment_dir / "best_model.pt")

        export_path = self.model.export(format=format, **kwargs)
        logger.info(f"Model exported to {export_path}")
        return export_path

    def predict(self, image_path: str, confidence: float = 0.25):
        if self.model is None:
            self.model = YOLO(self.experiment_dir / "best_model.pt")

        results = self.model(image_path, conf=confidence)
        return results