# !filepath: src/model/trainer_regression_frame1to14.py

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset.regression_dataset import RegressionDataset

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

    torch.save(model.state_dict(), output_dir / "regression_model.pt")
    print(f"[INFO] Model saved to {output_dir / 'regression_model.pt'}")

    test_csv = Path("data/test.csv")
    if test_csv.exists():
        test_ds = RegressionDataset(test_csv)
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
    import torchvision.utils as vutils
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    count = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            x = x * 2 - 1
            y = y * 2 - 1
            preds = model(x)

            for i in range(min(x.size(0), limit - count)):
                vutils.save_image((x[i] + 1) / 2, output_dir / f"input_{count}.png")
                vutils.save_image((y[i] + 1) / 2, output_dir / f"target_{count}.png")
                vutils.save_image((preds[i] + 1) / 2, output_dir / f"predicted_{count}.png")
                count += 1
                if count >= limit:
                    return

if __name__ == "__main__":
    train_regression(
        train_csv=Path("data/train.csv"),
        val_csv=Path("data/val.csv"),
        epochs=100,
        batch_size=32,
        lr=1e-3,
        output_dir=Path("outputs")
    )
