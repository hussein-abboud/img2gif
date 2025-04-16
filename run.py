"""
Module: run.py
Description: Project CLI for img2gif
"""

import argparse
import subprocess
from datetime import datetime
import os
import sys



def run_git_command(command: list[str]) -> None:
    try:
        subprocess.run(["git"] + command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Git command failed: {' '.join(command)}")
        print(e)


def git_init():
    if not os.path.exists(".git"):
        print("[GIT] Initializing git repository...")
        run_git_command(["init"])
    else:
        print("[GIT] Repository already initialized.")


def git_add_commit_push(msg: str):
    print("[GIT] Adding all changes...")
    run_git_command(["add", "."])

    print(f"[GIT] Committing with message: {msg}")
    run_git_command(["commit", "-m", msg])

    print("[GIT] Pushing to origin...")
    run_git_command(["push", "-u", "origin", "main"])


def main():
    parser = argparse.ArgumentParser(description="img2gif CLI")
    parser.add_argument("--simulate", action="store_true", help="Run simulation")
    parser.add_argument("--preprocess", action="store_true", help="Run preprocessing")
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--export", action="store_true", help="Export to GIF")
    parser.add_argument("--git-init", action="store_true", help="Initialize git")
    parser.add_argument("--git-commit", metavar="msg", help="Add, commit and push all files")
    parser.add_argument("--check-preprocessed", action="store_true", help="Check preprocessed .pt files for consistency")
    parser.add_argument("--index", action="store_true", help="Create dataset index CSV")
    parser.add_argument("--split", action="store_true", help="Split dataset into train/val/test CSVs")
    parser.add_argument("--git", nargs="?", const=True, help="Stage, commit, and push changes with optional commit message")

#comment
    args = parser.parse_args()
    
    if args.git:
        print("[GIT] Staging and pushing changes...")
    
        try:
            subprocess.run(["git", "add", "."], check=True)
    
            # Use custom message or timestamp
            if isinstance(args.git, str):
                commit_msg = args.git
            else:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                commit_msg = f"Auto-commit: {timestamp}"
    
            subprocess.run(["git", "commit", "-m", commit_msg], check=True)
            subprocess.run(["git", "push", "-u", "origin", "main"], check=True)
            print("[GIT] Push successful.")
        except subprocess.CalledProcessError as e:
            print(f"[GIT ERROR] {e}")


    
    if args.split:
        from src.preprocessing.split_dataset import split_dataset
        print("[SPLIT] Creating stratified train/val/test splits...")
        split_dataset()

    
    if args.index:
        from src.preprocessing.create_dataset_index import create_index
        print("[INDEX] Creating dataset_index.csv from raw + processed data...")
        create_index()


    if args.check_preprocessed:
        print("[CHECK] Verifying preprocessed tensors...")
        from src.preprocessing.preprocessor_check import check_all
        check_all(verbose=False)


    if args.simulate:
        print("[SIMULATION] Running...")
        from src.simulation.simulator import run_simulation
        run_simulation()

    if args.preprocess:
        print("[PREPROCESSING] Running...")
        from src.preprocessing.preprocessor import run_preprocessor
        run_preprocessor()

    if args.train:
        print("[TRAINING] Running...")
        # from src.model.trainer import train_model
        # train_model()

    if args.export:
        print("[EXPORT] Running...")
        # from src.export.gif_writer import export_gif
        # export_gif()

    if args.git_init:
        git_init()

    if args.git_commit:
        git_add_commit_push(args.git_commit)

    if not any(vars(args).values()):
        parser.print_help()


if __name__ == "__main__":
    main()
