"""
Module: run.py
Description: Command-line entry point for img2gif project
"""

import argparse
from src.simulation import simulator
from src.preprocessing import processor
from src.model import trainer
from src.export import gif_writer


def main():
    parser = argparse.ArgumentParser(description="img2gif project runner")
    parser.add_argument("--simulate", action="store_true", help="Run Pygame simulation")
    parser.add_argument("--preprocess", action="store_true", help="Run data preprocessing")
    parser.add_argument("--train", action="store_true", help="Train the AI model")
    parser.add_argument("--export", action="store_true", help="Export results to GIF/video")

    args = parser.parse_args()

    if args.simulate:
        print("[SIMULATION] Running simulation...")
        # simulator.run_simulation()

    if args.preprocess:
        print("[PREPROCESSING] Running data preprocessing...")
        # processor.run_preprocessing()

    if args.train:
        print("[TRAINING] Running model training...")
        # trainer.train_model()

    if args.export:
        print("[EXPORT] Generating GIFs/videos...")
        # gif_writer.export_gif()

    if not any(vars(args).values()):
        parser.print_help()


if __name__ == "__main__":
    main()
