import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
    TrainingArguments,
    Trainer as HFTrainer
)
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from base_trainer import BaseTrainer
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class TransformersDataset(Dataset):
    def __init__(self, texts: List[str], labels: Optional[List[int]] = None,
                 tokenizer=None, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        item = {key: val.squeeze(0) for key, val in encoding.items()}

        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


class TransformersTrainer(BaseTrainer):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = config.get("model_name", "bert-base-uncased")
        self.task = config.get("task", "sequence_classification")
        self.num_labels = config.get("num_labels", 2)
        self.max_length = config.get("max_length", 512)
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None

    def load_data(self) -> Tuple[DataLoader, DataLoader]:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        train_texts = self.config.get("train_texts", [])
        train_labels = self.config.get("train_labels", [])
        val_texts = self.config.get("val_texts", [])
        val_labels = self.config.get("val_labels", [])

        train_dataset = TransformersDataset(
            train_texts, train_labels, self.tokenizer, self.max_length
        )
        val_dataset = TransformersDataset(
            val_texts, val_labels, self.tokenizer, self.max_length
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.get("batch_size", 16),
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.get("batch_size", 16),
            shuffle=False
        )

        return train_loader, val_loader

    def build_model(self):
        if self.task == "sequence_classification":
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, num_labels=self.num_labels
            )
        elif self.task == "token_classification":
            model = AutoModelForTokenClassification.from_pretrained(
                self.model_name, num_labels=self.num_labels
            )
        elif self.task == "question_answering":
            model = AutoModelForQuestionAnswering.from_pretrained(self.model_name)
        else:
            raise ValueError(f"Unsupported task: {self.task}")

        model = model.to(self.device)

        self.optimizer = AdamW(
            model.parameters(),
            lr=self.config.get("learning_rate", 2e-5),
            eps=self.config.get("adam_epsilon", 1e-8)
        )

        return model

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in self.train_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            self.optimizer.zero_grad()
            outputs = self.model(**batch)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            if self.scheduler:
                self.scheduler.step()

            total_loss += loss.item()

            if self.task == "sequence_classification":
                predictions = torch.argmax(outputs.logits, dim=-1)
                correct += (predictions == batch['labels']).sum().item()
                total += batch['labels'].size(0)

        metrics = {"loss": total_loss / len(self.train_loader)}
        if self.task == "sequence_classification":
            metrics["accuracy"] = correct / total

        return metrics

    def validate(self) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)

                total_loss += outputs.loss.item()

                if self.task == "sequence_classification":
                    predictions = torch.argmax(outputs.logits, dim=-1)
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(batch['labels'].cpu().numpy())

        metrics = {"loss": total_loss / len(self.val_loader)}

        if self.task == "sequence_classification" and all_labels:
            metrics["accuracy"] = accuracy_score(all_labels, all_predictions)
            metrics["f1"] = f1_score(all_labels, all_predictions, average='weighted')
            precision, recall, _, _ = precision_recall_fscore_support(
                all_labels, all_predictions, average='weighted'
            )
            metrics["precision"] = precision
            metrics["recall"] = recall

        return metrics

    def train_with_hf_trainer(self):
        """Alternative training using Hugging Face Trainer"""
        train_loader, val_loader = self.load_data()
        self.model = self.build_model()

        training_args = TrainingArguments(
            output_dir=str(self.experiment_dir),
            num_train_epochs=self.config.get("epochs", 3),
            per_device_train_batch_size=self.config.get("batch_size", 16),
            per_device_eval_batch_size=self.config.get("batch_size", 16),
            warmup_steps=self.config.get("warmup_steps", 500),
            weight_decay=self.config.get("weight_decay", 0.01),
            logging_dir=str(self.experiment_dir / "logs"),
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model=self.config.get("monitor_metric", "loss"),
            greater_is_better=not self.config.get("minimize_metric", True),
            push_to_hub=self.config.get("push_to_hub", False),
        )

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return {
                'accuracy': accuracy_score(labels, predictions),
                'f1': f1_score(labels, predictions, average='weighted')
            }

        trainer = HFTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_loader.dataset,
            eval_dataset=val_loader.dataset,
            compute_metrics=compute_metrics if self.task == "sequence_classification" else None,
            tokenizer=self.tokenizer,
        )

        trainer.train()
        trainer.save_model(str(self.experiment_dir / "final_model"))

        return trainer

    def export_to_onnx(self, dummy_input: Optional[torch.Tensor] = None):
        if dummy_input is None:
            dummy_input = {
                'input_ids': torch.zeros(1, self.max_length, dtype=torch.long).to(self.device),
                'attention_mask': torch.ones(1, self.max_length, dtype=torch.long).to(self.device)
            }

        output_path = self.experiment_dir / "model.onnx"

        torch.onnx.export(
            self.model,
            tuple(dummy_input.values()),
            output_path,
            input_names=['input_ids', 'attention_mask'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence'},
                'attention_mask': {0: 'batch_size', 1: 'sequence'},
                'logits': {0: 'batch_size'}
            },
            opset_version=11
        )

        logger.info(f"Model exported to ONNX: {output_path}")
        return output_path