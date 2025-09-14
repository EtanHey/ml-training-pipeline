#!/usr/bin/env python3
"""
Interactive local training script with testing checkpoints
"""

import argparse
import yaml
import json
import sys
import os
from pathlib import Path
import logging
from typing import Dict, Any, Optional
import time
from datetime import datetime
import shutil

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from training.yolo_trainer import YOLOTrainer
from training.transformers_trainer import TransformersTrainer
from training.custom_trainer import CustomTrainer

# Rich for better CLI experience
try:
    from rich.console import Console
    from rich.prompt import Prompt, Confirm
    from rich.table import Table
    from rich.progress import track
    from rich import print as rprint
    console = Console()
except ImportError:
    console = None
    rprint = print
    Prompt = input
    Confirm = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InteractiveTrainer:
    """Interactive training with human-in-the-loop testing"""

    def __init__(self, model_type: str, config_path: str):
        self.model_type = model_type
        self.config = self.load_config(config_path)
        self.trainer = None
        self.test_results = []
        self.iteration = 0

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def save_config(self, config_path: str = None):
        """Save current configuration"""
        if config_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config_path = f"experiments/configs/config_{timestamp}.yaml"

        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

        rprint(f"[green]✓ Configuration saved to {config_path}[/green]")
        return config_path

    def get_trainer(self):
        """Get appropriate trainer based on model type"""
        trainers = {
            'yolo': YOLOTrainer,
            'transformers': TransformersTrainer,
            'custom': CustomTrainer
        }

        if self.model_type not in trainers:
            raise ValueError(f"Unknown model type: {self.model_type}")

        return trainers[self.model_type](self.config)

    def train_iteration(self):
        """Run one training iteration"""
        self.iteration += 1
        rprint(f"\n[bold blue]━━━ Training Iteration {self.iteration} ━━━[/bold blue]")

        # Display current configuration
        self.display_config()

        # Create trainer
        self.trainer = self.get_trainer()

        # Run training
        rprint("[yellow]🚀 Starting training...[/yellow]")
        try:
            if self.model_type == 'yolo':
                results = self.trainer.train()
            else:
                self.trainer.train()
            rprint("[green]✓ Training completed![/green]")
            return True
        except Exception as e:
            rprint(f"[red]✗ Training failed: {e}[/red]")
            return False

    def test_model(self):
        """Interactive model testing"""
        rprint("\n[bold cyan]━━━ Model Testing Phase ━━━[/bold cyan]")

        test_dir = Path("test_samples")
        test_dir.mkdir(exist_ok=True)

        rprint(f"[yellow]Place test images/data in: {test_dir.absolute()}[/yellow]")
        rprint("[yellow]Starting local test server...[/yellow]")

        # Start test server
        self.start_test_server()

        # Wait for user testing
        rprint("\n[bold]Test your model now![/bold]")
        rprint("- Web UI: http://localhost:7860")
        rprint(f"- Test samples: {test_dir.absolute()}")
        rprint("- Model location: " + str(self.trainer.experiment_dir / "best_model.pt"))

        satisfied = Prompt.ask(
            "\n[cyan]Are you satisfied with the model performance?[/cyan]",
            choices=["yes", "no", "needs_work"],
            default="needs_work"
        )

        if satisfied == "yes":
            return True
        elif satisfied == "no":
            return False
        else:
            return None  # Needs more work

    def start_test_server(self):
        """Start Gradio test server"""
        test_script = """
import gradio as gr
import sys
sys.path.insert(0, '.')
from deployment.huggingface.app import create_interface

# Set model path
import os
os.environ['MODEL_PATH'] = '{model_path}'
os.environ['MODEL_TYPE'] = '{model_type}'

app = create_interface()
app.launch(server_name='0.0.0.0', server_port=7860, share=False)
        """.format(
            model_path=str(self.trainer.experiment_dir / "best_model.pt"),
            model_type=self.model_type
        )

        with open("test_server.py", "w") as f:
            f.write(test_script)

        os.system("python test_server.py &")
        time.sleep(3)  # Wait for server to start

    def modify_config(self):
        """Interactive configuration modification"""
        rprint("\n[bold yellow]━━━ Modify Configuration ━━━[/bold yellow]")

        options = {
            "1": ("Learning Rate", "learning_rate", float),
            "2": ("Batch Size", "batch_size", int),
            "3": ("Epochs", "epochs", int),
            "4": ("Image Size", "image_size", int),
            "5": ("Data Path", "data_path", str),
            "6": ("Model Name", "model_name", str),
            "7": ("Advanced Options", None, None),
            "8": ("Back", None, None)
        }

        for key, (name, _, _) in options.items():
            if name != "Back":
                current = self.config.get(options[key][1], "Not set")
                rprint(f"{key}. {name}: [cyan]{current}[/cyan]")
            else:
                rprint(f"{key}. {name}")

        choice = Prompt.ask("Select option to modify")

        if choice in options and choice not in ["7", "8"]:
            name, key, type_func = options[choice]
            current = self.config.get(key, "")
            new_value = Prompt.ask(f"New value for {name}", default=str(current))

            try:
                self.config[key] = type_func(new_value)
                rprint(f"[green]✓ {name} updated to {new_value}[/green]")
            except ValueError:
                rprint(f"[red]✗ Invalid value for {name}[/red]")

        elif choice == "7":
            # Advanced YAML editing
            import tempfile
            import subprocess

            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(self.config, f, default_flow_style=False)
                temp_path = f.name

            editor = os.environ.get('EDITOR', 'nano')
            subprocess.call([editor, temp_path])

            with open(temp_path, 'r') as f:
                self.config = yaml.safe_load(f)

            os.unlink(temp_path)
            rprint("[green]✓ Configuration updated[/green]")

    def display_config(self):
        """Display current configuration"""
        if console:
            table = Table(title="Current Configuration")
            table.add_column("Parameter", style="cyan")
            table.add_column("Value", style="green")

            for key, value in self.config.items():
                table.add_row(key, str(value))

            console.print(table)
        else:
            rprint("\nCurrent Configuration:")
            for key, value in self.config.items():
                rprint(f"  {key}: {value}")

    def save_snapshot(self):
        """Save current model snapshot"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_dir = Path(f"snapshots/{self.model_type}_{timestamp}")
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        # Copy model
        shutil.copy(
            self.trainer.experiment_dir / "best_model.pt",
            snapshot_dir / "model.pt"
        )

        # Save config
        with open(snapshot_dir / "config.yaml", 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

        # Save test results
        with open(snapshot_dir / "test_results.json", 'w') as f:
            json.dump(self.test_results, f, indent=2)

        rprint(f"[green]✓ Snapshot saved to {snapshot_dir}[/green]")
        return snapshot_dir

    def run(self):
        """Main interactive training loop"""
        rprint("[bold green]━━━ Interactive ML Training Pipeline ━━━[/bold green]")
        rprint(f"Model Type: [cyan]{self.model_type}[/cyan]")

        while True:
            rprint("\n[bold]Options:[/bold]")
            rprint("1. Train Model")
            rprint("2. Test Model")
            rprint("3. Modify Configuration")
            rprint("4. Save Snapshot")
            rprint("5. Deploy Model")
            rprint("6. Exit")

            choice = Prompt.ask("Select action", choices=["1", "2", "3", "4", "5", "6"])

            if choice == "1":
                success = self.train_iteration()
                if success:
                    test_now = Prompt.ask("Test the model now?", choices=["yes", "no"], default="yes")
                    if test_now == "yes":
                        self.test_model()

            elif choice == "2":
                if self.trainer is None:
                    rprint("[red]No model trained yet![/red]")
                else:
                    result = self.test_model()
                    self.test_results.append({
                        "iteration": self.iteration,
                        "satisfied": result,
                        "timestamp": datetime.now().isoformat()
                    })

            elif choice == "3":
                self.modify_config()
                save = Prompt.ask("Save configuration?", choices=["yes", "no"], default="yes")
                if save == "yes":
                    self.save_config()

            elif choice == "4":
                if self.trainer:
                    self.save_snapshot()
                else:
                    rprint("[red]No model to save![/red]")

            elif choice == "5":
                if self.trainer:
                    self.deploy_model()
                else:
                    rprint("[red]No model to deploy![/red]")

            elif choice == "6":
                rprint("[yellow]Goodbye! 👋[/yellow]")
                break

    def deploy_model(self):
        """Deploy model to selected platform"""
        rprint("\n[bold cyan]━━━ Model Deployment ━━━[/bold cyan]")

        target = Prompt.ask(
            "Select deployment target",
            choices=["huggingface", "runpod", "export", "cancel"],
            default="huggingface"
        )

        if target == "huggingface":
            os.system(f"cd deployment/huggingface && python deploy.py --model-path {self.trainer.experiment_dir / 'best_model.pt'}")
        elif target == "runpod":
            os.system(f"cd deployment/runpod && bash deploy.sh")
        elif target == "export":
            export_format = Prompt.ask("Export format", choices=["onnx", "tflite", "torchscript"], default="onnx")
            if self.model_type == 'yolo':
                self.trainer.export_model(format=export_format)
            else:
                self.trainer.export_to_onnx()
            rprint(f"[green]✓ Model exported to {export_format}[/green]")


def main():
    parser = argparse.ArgumentParser(description='Interactive ML Training Pipeline')
    parser.add_argument('--model-type', type=str, default='yolo',
                       choices=['yolo', 'transformers', 'custom'],
                       help='Type of model to train')
    parser.add_argument('--config', type=str, default='experiments/configs/default.yaml',
                       help='Path to configuration file')

    args = parser.parse_args()

    # Create default config if it doesn't exist
    config_path = Path(args.config)
    if not config_path.exists():
        config_path.parent.mkdir(parents=True, exist_ok=True)
        default_config = {
            'model_name': 'yolov8n' if args.model_type == 'yolo' else 'bert-base-uncased',
            'epochs': 10,
            'batch_size': 16,
            'learning_rate': 0.001,
            'data_path': './datasets',
            'image_size': 640,
            'device': 'auto'
        }
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        rprint(f"[yellow]Created default config at {config_path}[/yellow]")

    trainer = InteractiveTrainer(args.model_type, args.config)
    trainer.run()


if __name__ == "__main__":
    main()