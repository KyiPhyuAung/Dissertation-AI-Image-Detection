from pathlib import Path
import torch

# =============================================================================
# Project Directories
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_ROOT / "datasets"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

# =============================================================================
# Hardware
# =============================================================================

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# =============================================================================
# Dataset
# =============================================================================

IMAGE_SIZE = 384
NUM_CLASSES = 2

# =============================================================================
# Training
# =============================================================================

BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 20
RANDOM_SEED = 42