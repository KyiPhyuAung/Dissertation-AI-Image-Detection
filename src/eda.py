from pathlib import Path
import pandas as pd
from PIL import Image

from src.config import DATASET_DIR, RESULTS_DIR


def analyze_dataset(dataset_name="sample"):
    dataset_path = DATASET_DIR / dataset_name
    results_path = RESULTS_DIR / f"{dataset_name}_eda.csv"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    for class_name in ["real", "fake"]:
        class_dir = dataset_path / class_name

        for image_path in class_dir.rglob("*"):
            if image_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                with Image.open(image_path) as img:
                    rows.append({
                        "filename": image_path.name,
                        "class": class_name,
                        "width": img.width,
                        "height": img.height,
                        "format": img.format,
                        "path": str(image_path)
                    })

    df = pd.DataFrame(rows)

    print("=" * 50)
    print(f"Dataset: {dataset_name}")
    print("=" * 50)
    print("Total images:", len(df))
    print("\nClass counts:")
    print(df["class"].value_counts())
    print("\nImage formats:")
    print(df["format"].value_counts())
    print("\nResolution summary:")
    print(df[["width", "height"]].describe())

    df.to_csv(results_path, index=False)
    print(f"\nSaved EDA results to: {results_path}")


if __name__ == "__main__":
    analyze_dataset("sample")