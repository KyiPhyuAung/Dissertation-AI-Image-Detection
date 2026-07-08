from pathlib import Path
from PIL import Image

from torch.utils.data import Dataset


class AIImageDataset(Dataset):
    """
    PyTorch Dataset for binary AI-generated image detection.

    Expected folder structure:

    dataset_root/
        real/
            image1.jpg
            image2.png
        fake/
            image3.jpg
            image4.png

    Labels:
        real = 0
        fake = 1
    """

    def __init__(self, dataset_root, transform=None):
        self.dataset_root = Path(dataset_root)
        self.transform = transform

        self.class_to_idx = {
            "real": 0,
            "fake": 1,
        }

        self.samples = []

        for class_name, label in self.class_to_idx.items():
            class_dir = self.dataset_root / class_name

            if not class_dir.exists():
                raise FileNotFoundError(f"Missing folder: {class_dir}")

            for image_path in class_dir.rglob("*"):
                if image_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                    self.samples.append((image_path, label))

        if len(self.samples) == 0:
            raise ValueError(f"No images found in {self.dataset_root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label