from src.dataset import AIImageDataset
from src.transforms import get_train_transforms
from src.config import DATASET_DIR

dataset = AIImageDataset(
    dataset_root=DATASET_DIR / "sample",
    transform=get_train_transforms()
)

print("=" * 40)
print("Dataset loaded successfully")
print("=" * 40)

print(f"Number of images: {len(dataset)}")

image, label = dataset[0]

print(f"Image shape: {image.shape}")
print(f"Image dtype: {image.dtype}")
print(f"Label: {label}")

print("=" * 40)