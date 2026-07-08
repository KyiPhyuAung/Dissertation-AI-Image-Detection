from src.config import TINY_GENIMAGE_DIR
from src.transforms import get_eval_transforms
from src.genimage_dataset import GenImageDataset

dataset = GenImageDataset(
    root_dir=TINY_GENIMAGE_DIR,
    generator="imagenet_midjourney",
    split="train",
    transform=get_eval_transforms()
)

print("=" * 50)
print("Tiny GenImage Test")
print("=" * 50)

print("Total images:", len(dataset))

image, label = dataset[0]

print("Image shape:", image.shape)
print("Label:", label)