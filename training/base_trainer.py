from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import os
import json
import logging
from datetime import datetime
import torch
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.device = self._setup_device()
        self.experiment_dir = self._setup_experiment_dir()
        self.metrics = {}

    def _setup_device(self) -> torch.device:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using Apple Silicon GPU (MPS)")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU")
        return device

    def _setup_experiment_dir(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = self.config.get("experiment_name", "experiment")
        exp_dir = Path(f"experiments/runs/{exp_name}_{timestamp}")
        exp_dir.mkdir(parents=True, exist_ok=True)

        with open(exp_dir / "config.json", "w") as f:
            json.dump(self.config, f, indent=2)

        logger.info(f"Experiment directory: {exp_dir}")
        return exp_dir

    @abstractmethod
    def load_data(self) -> Tuple[Any, Any]:
        pass

    @abstractmethod
    def build_model(self) -> Any:
        pass

    @abstractmethod
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        pass

    @abstractmethod
    def validate(self) -> Dict[str, float]:
        pass

    def train(self):
        logger.info("Starting training...")
        train_loader, val_loader = self.load_data()
        self.model = self.build_model()

        best_metric = float('inf') if self.config.get("minimize_metric", True) else -float('inf')

        for epoch in range(self.config["epochs"]):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()

            self._log_metrics(epoch, train_metrics, val_metrics)

            current_metric = val_metrics.get(self.config.get("monitor_metric", "loss"))
            if self._is_better(current_metric, best_metric):
                best_metric = current_metric
                self.save_checkpoint("best")
                logger.info(f"New best model saved: {self.config.get('monitor_metric', 'loss')} = {current_metric:.4f}")

            if (epoch + 1) % self.config.get("save_interval", 10) == 0:
                self.save_checkpoint(f"epoch_{epoch+1}")

    def _is_better(self, current: float, best: float) -> bool:
        if self.config.get("minimize_metric", True):
            return current < best
        return current > best

    def _log_metrics(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        metrics_entry = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": val_metrics,
            "timestamp": datetime.now().isoformat()
        }

        metrics_file = self.experiment_dir / "metrics.jsonl"
        with open(metrics_file, "a") as f:
            f.write(json.dumps(metrics_entry) + "\n")

        logger.info(f"Epoch {epoch}: Train - {train_metrics}, Val - {val_metrics}")

    def save_checkpoint(self, name: str = "checkpoint"):
        checkpoint_path = self.experiment_dir / f"{name}.pth"
        torch.save({
            "model_state_dict": self.model.state_dict() if hasattr(self.model, 'state_dict') else self.model,
            "config": self.config,
            "metrics": self.metrics
        }, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def export_to_onnx(self, dummy_input: torch.Tensor, output_path: Optional[str] = None):
        if output_path is None:
            output_path = self.experiment_dir / "model.onnx"

        self.model.eval()
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        logger.info(f"Model exported to ONNX: {output_path}")
        return output_path