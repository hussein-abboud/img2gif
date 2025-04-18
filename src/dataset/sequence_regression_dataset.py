# !filepath: src/dataset/sequence_regression_dataset.py

# Purpose:
# This dataset class loads preprocessed tensor files and returns a tuple of (input_frame, target_sequence),
# where the input is frame 0 and the target is frames 1 to 14. It is designed for training models that 
# predict future frames from a single initial frame in a video sequence.


import torch
from torch.utils.data import Dataset
from pathlib import Path
import csv

class SequenceRegressionDataset(Dataset):
    """
    Dataset for predicting frames 1 to 14 from frame 0.
    """

    def __init__(self, csv_file: Path):
        self.samples = []
        with open(csv_file, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append(Path(row["pt_path"]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        pt_path = self.samples[idx]
        tensor = torch.load(pt_path, weights_only=True)  # shape: (15, 3, 64, 64)
        frame_0 = tensor[0]              # (3, 64, 64)
        frames_1_to_14 = tensor[1:]      # (14, 3, 64, 64)
        return frame_0, frames_1_to_14
