# !filepath: src/model/trainer_sequence_regression.py

# Purpose:
# This script trains multiple sequence-to-sequence regression models for video frame prediction using
# a variety of UNet-based architectures. It supports model selection, training loop execution,
# loss tracking, validation, test evaluation, and GIF visualization of predictions.
# Loss plots and trained model weights are saved for each architecture variant.

import time
start = time.perf_counter()

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import imageio

from src.dataset.sequence_regression_dataset import SequenceRegressionDataset
from src.model.models_regression_sequence import (
    BaseUNetSequenceModel,
    SkipUNetSequenceModel,
    LSTMUNetSequenceModel,
    ResLSTMUNetSequenceModel,
    AttnUNetSequenceModel,
    ViTUNetSequenceModel
)
from src.losses.loss_composer import LossComposer
from src.utils.project_paths import TRAIN_CSV, VAL_CSV, TEST_CSV, OUTPUTS_DIR_SEQUENCE

criterion = LossComposer(
    use_l1=True,
    use_grad=False,
    use_perceptual=True,
    use_ssim=False
)

def train_sequence_model(
    nn_model: nn.Module,
    train_csv: Path,
    val_csv: Path,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    output_dir: Path = OUTPUTS_DIR_SEQUENCE,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    comment: str = "none"
):
    output_dir.mkdir(exist_ok=True, parents=True)

    train_ds = SequenceRegressionDataset(train_csv)
    val_ds = SequenceRegressionDataset(val_csv)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, pin_memory=True)

    model = nn_model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        val_loss = evaluate(model, val_loader, criterion, device)
        train_losses.append(avg_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.6f} | Val Loss: {val_loss:.6f}")

    torch.save(model.state_dict(), output_dir / f"{comment}.pt")
    print(f"[INFO] Model saved to {output_dir / f'{comment}.pt'}")

    plt.figure()
    plt.plot(range(1, epochs + 1), train_losses, label='Train')
    plt.plot(range(1, epochs + 1), val_losses, label='Val')
    plt.title("Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{comment}_loss_plot.png")
    plt.close()

    if TEST_CSV.exists():
        test_ds = SequenceRegressionDataset(TEST_CSV)
        test_loader = DataLoader(test_ds, batch_size=batch_size)
        test_loss = evaluate(model, test_loader, criterion, device)
        print(f"[TEST] Test Loss: {test_loss:.6f}")
        save_gif_outputs(model, test_loader, output_dir / f"test_gifs_{comment}", device, limit=50)
    else:
        print("[WARNING] No test.csv found – skipping test evaluation.")


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: str) -> float:
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def save_gif_outputs(model: nn.Module, loader: DataLoader, output_dir: Path, device: str, limit: int = 50):
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    counter = 0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            pred = model(x)
            pred = (pred + 1) / 2  # back to [0, 1]
            for i in range(pred.size(0)):
                frames = [pred[i, j].cpu().permute(1, 2, 0).numpy() for j in range(14)]
                frames = [(f * 255).astype("uint8") for f in frames]
                gif_path = output_dir / f"prediction_{counter}.gif"
                imageio.mimsave(gif_path, frames, duration=0.2)
                counter += 1
                if counter >= limit:
                    return


if __name__ == "__main__":
    n_epochs = 5
    train_sequence_model(BaseUNetSequenceModel(), TRAIN_CSV, VAL_CSV, n_epochs, 32, 1e-3, OUTPUTS_DIR_SEQUENCE, comment="BaseUNetSequenceModel")
    train_sequence_model(SkipUNetSequenceModel(), TRAIN_CSV, VAL_CSV, n_epochs, 32, 1e-3, OUTPUTS_DIR_SEQUENCE, comment="SkipUNetSequenceModel")
    train_sequence_model(LSTMUNetSequenceModel(), TRAIN_CSV, VAL_CSV, n_epochs, 32, 1e-3, OUTPUTS_DIR_SEQUENCE, comment="LSTMUNetSequenceModel")
    train_sequence_model(ResLSTMUNetSequenceModel(), TRAIN_CSV, VAL_CSV, n_epochs, 32, 1e-3, OUTPUTS_DIR_SEQUENCE, comment="ResLSTMUNetSequenceModel")
    train_sequence_model(AttnUNetSequenceModel(), TRAIN_CSV, VAL_CSV, n_epochs, 32, 1e-3, OUTPUTS_DIR_SEQUENCE, comment="AttnUNetSequenceModel")
    train_sequence_model(ViTUNetSequenceModel(), TRAIN_CSV, VAL_CSV, n_epochs, 32, 1e-3, OUTPUTS_DIR_SEQUENCE, comment="ViTUNetSequenceModel")

    end = time.perf_counter()
    print(f"Epoch time: {end - start:.2f}s")
