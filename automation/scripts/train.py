#!/usr/bin/env python3
"""
Automated training script for different model types
"""

import argparse
import yaml
import json
import sys
import os
from pathlib import Path
import logging
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.yolo_trainer import YOLOTrainer
from training.transformers_trainer import TransformersTrainer
from training.custom_trainer import CustomTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_trainer(model_type: str, config: Dict[str, Any]):
    """Get appropriate trainer based on model type"""
    trainers = {
        'yolo': YOLOTrainer,
        'transformers': TransformersTrainer,
        'custom': CustomTrainer
    }

    if model_type not in trainers:
        raise ValueError(f"Unknown model type: {model_type}")

    return trainers[model_type](config)


def main():
    parser = argparse.ArgumentParser(description='Train ML models')
    parser.add_argument('--model-type', type=str, required=True,
                       choices=['yolo', 'transformers', 'custom'],
                       help='Type of model to train')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--data-path', type=str,
                       help='Override data path from config')
    parser.add_argument('--epochs', type=int,
                       help='Override number of epochs')
    parser.add_argument('--batch-size', type=int,
                       help='Override batch size')
    parser.add_argument('--learning-rate', type=float,
                       help='Override learning rate')
    parser.add_argument('--output-dir', type=str,
                       help='Output directory for model and logs')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from checkpoint')
    parser.add_argument('--export-onnx', action='store_true',
                       help='Export model to ONNX format')
    parser.add_argument('--push-to-hub', action='store_true',
                       help='Push model to Hugging Face Hub')

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override config with command line arguments
    if args.data_path:
        config['data_path'] = args.data_path
    if args.epochs:
        config['epochs'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.learning_rate:
        config['learning_rate'] = args.learning_rate
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.resume:
        config['resume'] = args.resume
    if args.push_to_hub:
        config['push_to_hub'] = args.push_to_hub

    # Add model type to config
    config['model_type'] = args.model_type

    logger.info(f"Starting training with model type: {args.model_type}")
    logger.info(f"Configuration: {json.dumps(config, indent=2)}")

    try:
        # Get appropriate trainer
        trainer = get_trainer(args.model_type, config)

        # Run training
        if args.model_type == 'yolo':
            results = trainer.train()
        else:
            trainer.train()

        # Export to ONNX if requested
        if args.export_onnx:
            logger.info("Exporting model to ONNX format...")
            if args.model_type == 'yolo':
                onnx_path = trainer.export_model(format='onnx')
            else:
                onnx_path = trainer.export_to_onnx()
            logger.info(f"Model exported to: {onnx_path}")

        # Save model path for CI/CD pipeline
        model_path = trainer.experiment_dir / "best_model.pt"
        with open('/tmp/model_path.txt', 'w') as f:
            f.write(str(model_path))

        # Save metrics for CI/CD pipeline
        metrics = {
            'model_type': args.model_type,
            'experiment_dir': str(trainer.experiment_dir),
            'model_path': str(model_path)
        }

        if hasattr(trainer, 'metrics'):
            metrics.update(trainer.metrics)

        with open('/tmp/metrics.json', 'w') as f:
            json.dump(metrics, f)

        logger.info("Training completed successfully!")
        logger.info(f"Model saved to: {model_path}")
        logger.info(f"Experiment directory: {trainer.experiment_dir}")

        return 0

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())