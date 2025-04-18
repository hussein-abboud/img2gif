# !filepath: src/model/trainer_regression_frame1to14.py

# Purpose:
# This script trains a U-Net-based autoencoder model to regress directly from frame 0 to frame 14
# in a simulated video sequence. It is designed for frame-to-frame prediction where only the initial
# and final frames of a 15-frame sequence are used.
#
# Model Architecture:
# - The model is a U-Net-like encoder-decoder with residual blocks in the bottleneck and decoder.
# - It uses skip connections to retain spatial detail from earlier layers.
# - The final output is passed through a Tanh activation to match normalized pixel range [-1, 1].
#
# Training Workflow:
# - Loads data using the `RegressionDataset`, which returns (frame_0, frame_14) pairs from .pt files.
# - Trains using MSE loss and Adam optimizer.
# - Saves training/validation loss plots and model weights.
# - Optionally evaluates on a test set and saves sample predictions as images for visual inspection.
#
# Use Case:
# This is well-suited for fast inference of end-state visual predictions, especially when temporal
# intermediate states are not required or available.

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from datetime import datetime
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.dataset.regression_dataset import RegressionDataset
from src.utils.project_paths import TRAIN_CSV, VAL_CSV, TEST_CSV, OUTPUTS_DIR_SINGLEFRAME


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.residual(x)
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return self.relu(x + residual)


class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNetAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = UNetBlock(3, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = UNetBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = UNetBlock(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            ResidualBlock(128, 256),
            ResidualBlock(256, 256)
        )

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = ResidualBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = ResidualBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = ResidualBlock(64, 32)

        self.final = nn.Conv2d(32, 3, 1)
        self.activation = nn.Tanh()

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        b = self.bottleneck(self.pool3(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.activation(self.final(d1))


def train_regression(
    train_csv: Path,
    val_csv: Path,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    output_dir: Path = OUTPUTS_DIR_SINGLEFRAME,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    output_dir.mkdir(exist_ok=True, parents=True)

    train_ds = RegressionDataset(train_csv)
    val_ds = RegressionDataset(val_csv)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = UNetAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            x = x * 2 - 1
            y = y * 2 - 1

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

        if epoch == epochs - 1:
            save_sample_outputs(model, val_loader, output_dir, device)

    torch.save(model.state_dict(), output_dir / "regression_model.pt")
    print(f"[INFO] Model saved to {output_dir / 'regression_model.pt'}")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_losses, label="Train Loss", linewidth=2)
    plt.plot(range(1, epochs + 1), val_losses, label="Val Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss History ({datetime.now().strftime('%Y-%m-%d')}) | Epochs={epochs} | Batch={batch_size} | LR={lr}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "loss_history.png")
    plt.close()

    if TEST_CSV.exists():
        test_ds = RegressionDataset(TEST_CSV)
        test_loader = DataLoader(test_ds, batch_size=batch_size)
        test_loss = evaluate(model, test_loader, criterion, device)
        print(f"[TEST] Test Loss: {test_loss:.6f}")
        save_sample_outputs(model, test_loader, output_dir / "test_examples", device, limit=100)
    else:
        print("[WARNING] No test.csv found – skipping test evaluation.")


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: str) -> float:
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            x = x * 2 - 1
            y = y * 2 - 1
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def save_sample_outputs(model: nn.Module, loader: DataLoader, output_dir: Path, device: str, limit: int = 8):
    import torch
    import torchvision.utils as vutils

    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    samples = []

    with torch.no_grad():
        count = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            x = x * 2 - 1
            y = y * 2 - 1
            preds = model(x)

            for i in range(x.size(0)):
                samples.append((x[i], y[i], preds[i]))
                count += 1
                if count >= limit:
                    break
            if count >= limit:
                break

    max_rows = 5
    image_width = samples[0][0].shape[2]
    image_height = samples[0][0].shape[1]
    channels = samples[0][0].shape[0]

    blank_image = torch.zeros(channels, image_height, image_width, device=device)
    header_text_images = {
        "Input": torch.full_like(blank_image, 0.8),
        "Target": torch.full_like(blank_image, 0.8),
        "Predicted": torch.full_like(blank_image, 0.8),
    }

    def create_grid_page(samples_chunk: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]], file_index: int):
        row_images = []
        row_images.extend([header_text_images["Input"], header_text_images["Target"], header_text_images["Predicted"]])
        for input_img, target_img, pred_img in samples_chunk:
            row_images.extend([
                (input_img + 1) / 2,
                (target_img + 1) / 2,
                (pred_img + 1) / 2,
            ])
        for _ in range(max_rows - len(samples_chunk)):
            row_images.extend([blank_image, blank_image, blank_image])
        grid = vutils.make_grid([img.cpu() for img in row_images], nrow=3, padding=4)
        vutils.save_image(grid, output_dir / f"samples_{file_index}.png")

    for i in range(0, len(samples), max_rows):
        chunk = samples[i:i + max_rows]
        create_grid_page(chunk, file_index=i // max_rows)


if __name__ == "__main__":
    train_regression(
        train_csv=TRAIN_CSV,
        val_csv=VAL_CSV,
        epochs=100,
        batch_size=32,
        lr=1e-3,
        output_dir=OUTPUTS_DIR_SINGLEFRAME
    )
