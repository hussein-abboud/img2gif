# !filepath: src/model/trainer_regression_frame1to14.py
# Unet

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset.regression_dataset import RegressionDataset

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
        # Encoder
        self.enc1 = UNetBlock(3, 32)
        self.pool1 = nn.MaxPool2d(2)  # 32x32
        self.enc2 = UNetBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2)  # 16x16
        self.enc3 = UNetBlock(64, 128)
        self.pool3 = nn.MaxPool2d(2)  # 8x8

        # Bottleneck
        self.bottleneck = UNetBlock(128, 256)

        # Decoder
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = UNetBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = UNetBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = UNetBlock(64, 32)

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
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    output_dir: Path = Path("outputs"),
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

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.6f} | Val Loss: {val_loss:.6f}")

        if epoch == epochs - 1:
            save_sample_outputs(model, val_loader, output_dir, device)

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

def save_sample_outputs(model: nn.Module, loader: DataLoader, output_dir: Path, device: str):
    import torchvision.utils as vutils
    model.eval()
    with torch.no_grad():
        x, y = next(iter(loader))
        x, y = x.to(device), y.to(device)
        x = x * 2 - 1
        y = y * 2 - 1
        preds = model(x)
        vutils.save_image((preds + 1) / 2, output_dir / "predicted.png", nrow=4)
        vutils.save_image((y + 1) / 2, output_dir / "target.png", nrow=4)
        vutils.save_image((x + 1) / 2, output_dir / "input.png", nrow=4)

if __name__ == "__main__":
    train_regression(
        train_csv=Path("data/train.csv"),
        val_csv=Path("data/val.csv"),
        epochs=10,
        batch_size=32,
        lr=1e-3,
        output_dir=Path("outputs")
    )