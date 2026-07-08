from PIL import Image
import pandas as pd

from src.config import TINY_GENIMAGE_DIR, RESULTS_DIR
from src.genimage_dataset import GenImageDataset


def main():
    generator = "imagenet_midjourney"
    split = "train"

    dataset = GenImageDataset(
        root_dir=TINY_GENIMAGE_DIR,
        generator=generator,
        split=split,
        transform=None
    )

    rows = []

    for image_path, label in dataset.samples:
        with Image.open(image_path) as img:
            rows.append({
                "path": str(image_path),
                "class": "fake" if label == 1 else "real",
                "width": img.width,
                "height": img.height,
                "format": img.format
            })

    df = pd.DataFrame(rows)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"{generator}_{split}_eda.csv"
    df.to_csv(output_path, index=False)

    print("EDA completed.")
    print("=" * 50)
    print("Generator:", generator)
    print("Split:", split)
    print("Total images:", len(df))
    print("\nClass counts:")
    print(df["class"].value_counts())
    print("\nFormats:")
    print(df["format"].value_counts())
    print("\nResolution summary:")
    print(df[["width", "height"]].describe())
    print("\nSaved to:", output_path)


if __name__ == "__main__":
    main()