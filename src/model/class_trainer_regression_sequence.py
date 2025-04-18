# !filepath: src/model/class_trainer_sequence_regression.py

"""
Purpose:
This module defines the `SequenceTrainer` class, which encapsulates the logic for training sequence-to-sequence models
on video frame prediction tasks. It supports:
- Dataset loading and preprocessing
- Training loop with MSE loss and Adam optimizer
- Validation and test evaluation
- Model checkpointing and loss plotting
- Saving predicted sequences as animated GIFs for visual inspection

Use Case:
Useful for training UNet-based architectures to predict sequences of simulation frames based on the initial input.
This modular class allows easy integration with CLIs, experiments, or pipelines.
"""

from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import imageio

from src.dataset.sequence_regression_dataset import SequenceRegressionDataset
from src.utils.project_paths import TRAIN_CSV, VAL_CSV, TEST_CSV, OUTPUTS_DIR_SEQUENCE
from src.losses.loss_composer import LossComposer


class SequenceTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_csv: Path = TRAIN_CSV,
        val_csv: Path = VAL_CSV,
        test_csv: Optional[Path] = TEST_CSV,
        output_dir: Path = OUTPUTS_DIR_SEQUENCE,
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 1e-3,
        comment: str = "unnamed",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.test_csv = test_csv
        self.output_dir = output_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.comment = comment
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def run(self):
        self.output_dir.mkdir(exist_ok=True, parents=True)

        train_loader = DataLoader(
            SequenceRegressionDataset(self.train_csv),
            batch_size=self.batch_size, shuffle=True, pin_memory=True
        )
        val_loader = DataLoader(
            SequenceRegressionDataset(self.val_csv),
            batch_size=self.batch_size, pin_memory=True
        )

        train_losses, val_losses = [], []

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                out = self.model(x)
                loss = self.criterion(out, y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * x.size(0)

            avg_loss = total_loss / len(train_loader.dataset)
            val_loss = self._evaluate(val_loader)

            train_losses.append(avg_loss)
            val_losses.append(val_loss)

            print(f"Epoch {epoch+1}/{self.epochs} | Train Loss: {avg_loss:.6f} | Val Loss: {val_loss:.6f}")

        self._save_model()
        self._plot_losses(train_losses, val_losses)

        if self.test_csv and self.test_csv.exists():
            test_loader = DataLoader(
                SequenceRegressionDataset(self.test_csv),
                batch_size=self.batch_size
            )
            test_loss = self._evaluate(test_loader)
            print(f"[TEST] Test Loss: {test_loss:.6f}")
            self._save_gifs(test_loader)

    def _evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x)
                loss = self.criterion(out, y)
                total_loss += loss.item() * x.size(0)
        return total_loss / len(loader.dataset)

    def _save_model(self):
        path = self.output_dir / f"{self.comment}.pt"
        torch.save(self.model.state_dict(), path)
        print(f"[INFO] Model saved to {path}")

    def _plot_losses(self, train_losses, val_losses):
        plt.figure()
        plt.plot(range(1, self.epochs + 1), train_losses, label='Train')
        plt.plot(range(1, self.epochs + 1), val_losses, label='Val')
        plt.title("Loss History")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{self.comment}_loss_plot.png")
        plt.close()

    def _save_gifs(self, loader: DataLoader, limit: int = 50):
        out_dir = self.output_dir / f"test_gifs_{self.comment}"
        out_dir.mkdir(parents=True, exist_ok=True)

        self.model.eval()
        counter = 0
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(self.device)
                pred = self.model(x)
                pred = (pred + 1) / 2  # back to [0, 1]
                for i in range(pred.size(0)):
                    frames = [pred[i, j].cpu().permute(1, 2, 0).numpy() for j in range(14)]
                    frames = [(f * 255).astype("uint8") for f in frames]
                    imageio.mimsave(out_dir / f"prediction_{counter}.gif", frames, duration=0.2)
                    counter += 1
                    if counter >= limit:
                        return
