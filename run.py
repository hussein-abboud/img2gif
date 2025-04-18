#!filepath: run.py

"""
Module: run.py
Description: Unified CLI entrypoint for the img2gif pipeline.

Purpose:
This script provides a command-line interface (CLI) to run the various stages
of the img2gif machine learning pipeline. It supports running simulations,
preprocessing image data, creating datasets, training models, listing available
model architectures, and managing git commits for version control.

Usage Examples:

1. Simulate data and process:
   $ python run.py simulate --n 500 --fraction 0.2

2. Train a specific model:
   $ python run.py train --model ResLSTMUNetSequenceModel --epochs 150

3. List all available model classes:
   $ python run.py list-models

4. Commit and push to Git:
   $ python run.py git --msg "Commit message here"

This CLI ensures the pipeline can be modularly triggered and makes it easier
for users to chain simulation, processing, and training tasks from the terminal.
"""

import argparse
from pathlib import Path
from datetime import datetime
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"

sys.path.append(str(SRC_ROOT))

from src.model.class_trainer_regression_sequence import SequenceTrainer

def run_simulation(n_runs: int):
    from src.simulation.generate_dataset import run_and_capture_simulation
    print(f"[SIMULATE] Running {n_runs} simulations...")
    if n_runs <= 10:
        n_runs =100
        print("100 sample points are minimum, 100 are being generated!")
    run_and_capture_simulation(n_runs=n_runs)

    from src.preprocessing.preprocessor import run_preprocessor
    print("[PREPROCESS] Processing raw frames into tensors...")
    run_preprocessor()

    from src.preprocessing.preprocessor_check import check_all
    print("[CHECK] Validating preprocessed tensor files...")
    check_all(verbose=False)

def create_index_and_split(fraction: float):
    from src.preprocessing.create_dataset_index import create_index
    print("[INDEX] Creating dataset_index.csv...")
    create_index()

    from src.preprocessing.split_dataset import split_dataset
    print(f"[SPLIT] Creating stratified train/val/test using {fraction*100:.0f}% of dataset...")
    split_dataset(fraction_of_data_to_be_used=fraction)

def list_models():
    import inspect
    from src.model import models_regression_sequence
    import torch.nn as nn

    model_classes = [
        name for name, cls in inspect.getmembers(models_regression_sequence, inspect.isclass)
        if issubclass(cls, nn.Module) and name.endswith("Model")
    ]

    print("Available Models:")
    for name in sorted(model_classes):
        print(f" - {name}")

def train_model(model_name: str, epochs: int):
    from src.model import models_regression_sequence
    import inspect

    cls = getattr(models_regression_sequence , model_name, None)
    if cls is None:
        print(f"[ERROR] Model '{model_name}' not found.")
        list_models()
        return

    trainer = SequenceTrainer(
        model=cls(),
        epochs=epochs,
        comment=model_name
    )
    trainer.run()

def main():
    parser = argparse.ArgumentParser(description="img2gif Pipeline CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Simulate
    simulate_parser = subparsers.add_parser("simulate", help="Run simulations and preprocessing")
    simulate_parser.add_argument("--n", type=int, default=1000, help="Number of simulations to run")
    simulate_parser.add_argument("--fraction", type=float, default=1.0,
                                 help="Fraction of data to use in dataset splitting (default=1.0)")

    # Train
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--model", required=True, help="Model class name to train")
    train_parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")

    # Models
    subparsers.add_parser("list-models", help="List available models")

    # Git commit
    git_parser = subparsers.add_parser("git", help="Commit and push changes to main")
    git_parser.add_argument("--msg", type=str, nargs="?", const=None,
                            help="Optional commit message (default: timestamp)")

    args = parser.parse_args()

    if args.command == "simulate":
        run_simulation(n_runs=args.n)
        create_index_and_split(fraction=args.fraction)

    elif args.command == "train":
        train_model(model_name=args.model, epochs=args.epochs)

    elif args.command == "list-models":
        list_models()

    elif args.command == "git":
        print("[GIT] Staging and pushing changes...")
        try:
            subprocess.run(["git", "add", "."], check=True)
            if args.msg:
                commit_msg = args.msg
            else:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                commit_msg = f"Auto-commit: {timestamp}"
            subprocess.run(["git", "commit", "-m", commit_msg], check=True)
            subprocess.run(["git", "push", "-u", "origin", "main"], check=True)
            print("[GIT] Push successful.")
        except subprocess.CalledProcessError as e:
            print(f"[GIT ERROR] {e}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()