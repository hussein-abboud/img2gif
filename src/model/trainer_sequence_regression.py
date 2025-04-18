# !filepath: src/model/trainer_sequence_regression.py
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
from src.dataset.sequence_regression_dataset import SequenceRegressionDataset
from src.model.sequence_regressor_models import simple_UNetSequenceModel
from src.model.sequence_regressor_models import skip_UNetSequenceModel
from src.model.sequence_regressor_models import ConvLSTM_UNetSequenceModel
from src.model.sequence_regressor_models import ResConvLSTMUNetV2
from src.model.sequence_regressor_models import AttentionUNetV3
from src.model.sequence_regressor_models import ViTUNetV4
import imageio

from src.losses.loss_composer import LossComposer

criterion = LossComposer(
    use_l1=True,
    use_grad=False,
    use_perceptual=True,
    use_ssim=False  # Enable if you have pytorch_msssim
)



def train_sequence_model(
    nn_model: nn.Module,     
    train_csv: Path,
    val_csv: Path,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    output_dir: Path = Path("outputs/sequence"),
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
            # Already normalized in dataset
            pass

            optimizer.zero_grad()
            out = model(x)
#
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        val_loss = evaluate(model, val_loader, criterion, device)
        train_losses.append(avg_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.6f} | Val Loss: {val_loss:.6f}")

    # Save model and loss plot
    torch.save(model.state_dict(), output_dir / "{}.pt".format(comment))
    print(f"[INFO] Model saved to {output_dir / '{}.pt'.format(comment)}")

    plt.figure()
    plt.plot(range(1, epochs + 1), train_losses, label='Train')
    plt.plot(range(1, epochs + 1), val_losses, label='Val')
    plt.title("Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "{}_loss_plot.png".format(comment))
    plt.close()

    test_csv = Path("data/test.csv")
    if test_csv.exists():
        test_ds = SequenceRegressionDataset(test_csv)
        test_loader = DataLoader(test_ds, batch_size=batch_size)
        test_loss = evaluate(model, test_loader, criterion, device)
        print(f"[TEST] Test Loss: {test_loss:.6f}")
        save_gif_outputs(model, test_loader, output_dir / "test_gifs_{}".format(comment), device, limit=50)
    else:
        print("[WARNING] No test.csv found – skipping test evaluation.")

def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: str) -> float:
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            # Already normalized in dataset
            pass
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)

def save_gif_outputs(model: nn.Module, loader: DataLoader, output_dir: Path, device: str, limit: int = 50):
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    counter = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            #x = x * 2 - 1
            pred = model(x)  # shape: (B, 14, 3, 64, 64)
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
    n_epochs=150
    train_sequence_model(
        simple_UNetSequenceModel(),
        train_csv=Path("data/train.csv"),
        val_csv=Path("data/val.csv"),
        epochs=n_epochs,
        batch_size=32,
        lr=1e-3,
        output_dir=Path("outputs/sequence/"), 
        comment="simple_UNetSequenceModel"
    )
    train_sequence_model(
        skip_UNetSequenceModel(),
        train_csv=Path("data/train.csv"),
        val_csv=Path("data/val.csv"),
        epochs=n_epochs,
        batch_size=32,
        lr=1e-3,
        output_dir=Path("outputs/sequence/"), 
        comment="skip_UNetSequenceModel"
    )
    train_sequence_model(
        ConvLSTM_UNetSequenceModel(),
        train_csv=Path("data/train.csv"),
        val_csv=Path("data/val.csv"),
        epochs=n_epochs,
        batch_size=32,
        lr=1e-3,
        output_dir=Path("outputs/sequence/"), 
        comment="ConvLSTM_UNetSequenceModel"
    )
    train_sequence_model(
        ResConvLSTMUNetV2(),
        train_csv=Path("data/train.csv"),
        val_csv=Path("data/val.csv"),
        epochs=n_epochs,
        batch_size=32,
        lr=1e-3,
        output_dir=Path("outputs/sequence/"), 
        comment="ResConvLSTMUNetV2"
    )
    train_sequence_model(
        AttentionUNetV3(),
        train_csv=Path("data/train.csv"),
        val_csv=Path("data/val.csv"),
        epochs=n_epochs,
        batch_size=32,
        lr=1e-3,
        output_dir=Path("outputs/sequence/"), 
        comment="AttentionUNetV3"
    )
    train_sequence_model(
        ViTUNetV4(),
        train_csv=Path("data/train.csv"),
        val_csv=Path("data/val.csv"),
        epochs=n_epochs,
        batch_size=32,
        lr=1e-3,
        output_dir=Path("outputs/sequence/"), 
        comment="ViTUNetV4"
    ) 

end = time.perf_counter()
print(f"Epoch time: {end - start:.2f}s")