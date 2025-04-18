# Purpose:
# This script preprocesses raw simulation frame sequences by resizing, normalizing, and stacking them into
# tensors. Each resulting tensor, representing 15 frames of a simulation, is saved as a `.pt` file in the
# processed data directory. This enables efficient loading for training and evaluation tasks.

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import os
import torch
from torchvision import transforms
from PIL import Image
from itertools import islice

from src.utils.project_paths import RAW_DATA_DIR as RAW_SIMULATION_PATH, PROCESSED_DATA_DIR as SAVE_PATH

IMG_SIZE = 64

def load_frames(frame_dir: Path) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])
    frames = []
    for i in range(15):
        img_path = frame_dir / f"frame_{i:02d}.png"
        if not img_path.exists():
            raise FileNotFoundError(f"Missing frame: {img_path}")
        image = Image.open(img_path).convert("RGB")
        frames.append(transform(image))
    return torch.stack(frames)  # shape: (15, 3, 64, 64)

def run_preprocessor():
    SAVE_PATH.mkdir(parents=True, exist_ok=True)

    # sample_dirs = islice(RAW_SIMULATION_PATH.iterdir(), 10)  # Only first 10 dirs
    sample_dirs = RAW_SIMULATION_PATH.iterdir()
    for sample_dir in sample_dirs:
        if not sample_dir.is_dir():
            continue

        try:
            tensor = load_frames(sample_dir)
            torch.save(tensor, SAVE_PATH / f"{sample_dir.name}.pt")
            print(f"Processed {sample_dir.name}")
        except Exception as e:
            print(f"[ERROR] Skipped {sample_dir.name}: {e}")
