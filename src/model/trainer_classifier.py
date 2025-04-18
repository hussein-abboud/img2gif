# !filepath: src/model/trainer.py

# Purpose:
# This script trains a simple convolutional neural network (CNN) for binary classification
# on simulated frame data. Each sample consists of the first frame of a simulation sequence
# and a binary label indicating whether the outcome was a "hit" or "miss".
#
# Model Architecture:
# - SimpleCNN: A basic CNN with two convolutional layers followed by two fully connected layers.
# - Designed for low-resource, fast classification tasks.
#
# Training Workflow:
# - Uses the `ClassifierDataset` to load (frame_0, label) pairs from .pt files and CSV metadata.
# - Optimizes cross-entropy loss using the Adam optimizer.
# - Evaluates accuracy on training, validation, and test sets after each epoch.
#
# Use Case:
# Ideal for outcome classification models that predict from static visual input (e.g., whether an AA
# defense will succeed or fail based on the first simulation frame). Useful as a lightweight baseline.


import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset.classifier_dataset import ClassifierDataset
from src.utils.project_paths import TRAIN_CSV, VAL_CSV, TEST_CSV


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def train_classifier(
    train_csv: Path,
    val_csv: Path,
    test_csv: Path,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    if not train_csv.exists() or not val_csv.exists() or not test_csv.exists():
        raise FileNotFoundError("One or more CSV files not found.")

    train_ds = ClassifierDataset(train_csv)
    val_ds = ClassifierDataset(val_csv)
    test_ds = ClassifierDataset(test_csv)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)

        train_acc = correct / total
        val_acc = evaluate(model, val_loader, device)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {total_loss/total:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    test_acc = evaluate(model, test_loader, device)
    print(f"\n[TEST] Accuracy on test set: {test_acc:.4f}")

def evaluate(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0

if __name__ == "__main__":
    train_classifier(
        train_csv=TRAIN_CSV,
        val_csv=VAL_CSV,
        test_csv=TEST_CSV,
        epochs=10,
        batch_size=32,
        lr=1e-3
    )
