import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
from base_trainer import BaseTrainer
import logging
from pathlib import Path
from torchvision import transforms, datasets
import timm

logger = logging.getLogger(__name__)


class CustomDataset(Dataset):
    def __init__(self, data_path: str, transform=None, mode='train'):
        self.data_path = Path(data_path)
        self.transform = transform
        self.mode = mode
        self.samples = []
        self.labels = []
        self._load_data()

    def _load_data(self):
        pass

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, label


class CustomModel(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.backbone = self._create_backbone()
        self.classifier = self._create_classifier()

    def _create_backbone(self):
        model_name = self.config.get("backbone", "resnet50")
        if model_name.startswith("timm/"):
            model = timm.create_model(
                model_name.replace("timm/", ""),
                pretrained=self.config.get("pretrained", True),
                num_classes=0
            )
        else:
            model = torch.hub.load(
                'pytorch/vision:v0.10.0',
                model_name,
                pretrained=self.config.get("pretrained", True)
            )
            if hasattr(model, 'fc'):
                in_features = model.fc.in_features
                model.fc = nn.Identity()
            elif hasattr(model, 'classifier'):
                in_features = model.classifier[-1].in_features
                model.classifier = nn.Identity()

        return model

    def _create_classifier(self):
        in_features = self.config.get("hidden_size", 2048)
        num_classes = self.config.get("num_classes", 10)
        dropout = self.config.get("dropout", 0.5)

        return nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output


class CustomTrainer(BaseTrainer):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None

    def load_data(self) -> Tuple[DataLoader, DataLoader]:
        transform_train = self._get_transforms('train')
        transform_val = self._get_transforms('val')

        if self.config.get("dataset_type") == "imagefolder":
            train_dataset = datasets.ImageFolder(
                self.config["train_path"],
                transform=transform_train
            )
            val_dataset = datasets.ImageFolder(
                self.config["val_path"],
                transform=transform_val
            )
        else:
            train_dataset = CustomDataset(
                self.config["train_path"],
                transform=transform_train,
                mode='train'
            )
            val_dataset = CustomDataset(
                self.config["val_path"],
                transform=transform_val,
                mode='val'
            )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.get("batch_size", 32),
            shuffle=True,
            num_workers=self.config.get("num_workers", 4),
            pin_memory=True if torch.cuda.is_available() else False
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.get("batch_size", 32),
            shuffle=False,
            num_workers=self.config.get("num_workers", 4),
            pin_memory=True if torch.cuda.is_available() else False
        )

        return self.train_loader, self.val_loader

    def _get_transforms(self, mode: str):
        if mode == 'train':
            return transforms.Compose([
                transforms.RandomResizedCrop(self.config.get("image_size", 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.4, 0.4, 0.4),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.config.get("normalize_mean", [0.485, 0.456, 0.406]),
                    std=self.config.get("normalize_std", [0.229, 0.224, 0.225])
                )
            ])
        else:
            return transforms.Compose([
                transforms.Resize(self.config.get("image_size", 224) + 32),
                transforms.CenterCrop(self.config.get("image_size", 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.config.get("normalize_mean", [0.485, 0.456, 0.406]),
                    std=self.config.get("normalize_std", [0.229, 0.224, 0.225])
                )
            ])

    def build_model(self) -> nn.Module:
        model = CustomModel(self.config)
        model = model.to(self.device)

        self.criterion = self._get_loss_function()

        self.optimizer = self._get_optimizer(model.parameters())

        self.scheduler = self._get_scheduler()

        return model

    def _get_loss_function(self):
        loss_fn = self.config.get("loss_function", "cross_entropy")
        if loss_fn == "cross_entropy":
            return nn.CrossEntropyLoss()
        elif loss_fn == "bce":
            return nn.BCEWithLogitsLoss()
        elif loss_fn == "mse":
            return nn.MSELoss()
        elif loss_fn == "focal":
            from focal_loss import FocalLoss
            return FocalLoss(gamma=self.config.get("focal_gamma", 2.0))
        else:
            raise ValueError(f"Unknown loss function: {loss_fn}")

    def _get_optimizer(self, parameters):
        opt_name = self.config.get("optimizer", "adam")
        lr = self.config.get("learning_rate", 1e-3)
        weight_decay = self.config.get("weight_decay", 1e-4)

        if opt_name == "adam":
            return optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
        elif opt_name == "adamw":
            return optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
        elif opt_name == "sgd":
            return optim.SGD(
                parameters, lr=lr, momentum=self.config.get("momentum", 0.9),
                weight_decay=weight_decay
            )
        elif opt_name == "rmsprop":
            return optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")

    def _get_scheduler(self):
        scheduler_name = self.config.get("scheduler", "cosine")
        if scheduler_name == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get("epochs", 100),
                eta_min=self.config.get("min_lr", 1e-6)
            )
        elif scheduler_name == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get("step_size", 30),
                gamma=self.config.get("gamma", 0.1)
            )
        elif scheduler_name == "exponential":
            return optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=self.config.get("gamma", 0.95)
            )
        elif scheduler_name == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min' if self.config.get("minimize_metric", True) else 'max',
                factor=self.config.get("factor", 0.1),
                patience=self.config.get("patience", 10)
            )
        else:
            return None

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            if batch_idx % self.config.get("log_interval", 10) == 0:
                logger.debug(
                    f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(self.train_loader.dataset)} '
                    f'({100. * batch_idx / len(self.train_loader):.0f}%)]\tLoss: {loss.item():.6f}'
                )

        if self.scheduler and not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step()

        return {
            "loss": total_loss / len(self.train_loader),
            "accuracy": correct / total,
            "lr": self.optimizer.param_groups[0]['lr']
        }

    def validate(self) -> Dict[str, float]:
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.criterion(output, target).item()

                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        val_loss /= len(self.val_loader)
        accuracy = correct / total

        if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(val_loss)

        return {
            "loss": val_loss,
            "accuracy": accuracy
        }

    def export_to_torchscript(self, output_path: Optional[str] = None):
        if output_path is None:
            output_path = self.experiment_dir / "model.pt"

        self.model.eval()
        example_input = torch.randn(
            1, 3,
            self.config.get("image_size", 224),
            self.config.get("image_size", 224)
        ).to(self.device)

        traced_model = torch.jit.trace(self.model, example_input)
        traced_model.save(str(output_path))

        logger.info(f"Model exported to TorchScript: {output_path}")
        return output_path