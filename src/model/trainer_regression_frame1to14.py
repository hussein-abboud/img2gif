# !filepath: src/model/trainer_regression_frame1to14.py

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset.regression_dataset import RegressionDataset


class SimpleAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),  # -> (32, 32, 32)
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # -> (64, 16, 16)
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # -> (128, 8, 8)
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # -> (64, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # -> (32, 32, 32)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),    # -> (3, 64, 64)
            nn.Tanh()  # normalize output to [-1, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x

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

    model = SimpleAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            x = x * 2 - 1  # normalize input to [-1, 1]
            y = y * 2 - 1  # normalize target to [-1, 1]

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
