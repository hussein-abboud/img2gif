# !filepath: src/utils/project_paths.py

from pathlib import Path

# Project root directory
ROOT = Path(__file__).resolve().parents[2]

# Data directories
DATA_DIR = ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INDEX_FILE = DATA_DIR / "dataset_index.csv"

# Dataset splits
TRAIN_CSV = DATA_DIR / "train.csv"
VAL_CSV = DATA_DIR / "val.csv"
TEST_CSV = DATA_DIR / "test.csv"

# Output directory
OUTPUTS_DIR = ROOT / "outputs"
OUTPUTS_DIR_SINGLEFRAME = ROOT / "outputs_singleframe"
OUTPUTS_DIR_SEQUENCE = ROOT / "outputs_sequence"

# INFER_APP
MODEL_DIR = ROOT / "outputs_sequence"  
GROUND_TRUTH_DIR = DATA_DIR / "raw"

# Project variables
IMG_SIZE = 64

