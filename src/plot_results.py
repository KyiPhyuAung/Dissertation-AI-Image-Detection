import pandas as pd
import matplotlib.pyplot as plt

from src.config import RESULTS_DIR, FIGURES_DIR


def clean_name(name):
    return (
        name.replace("imagenet_", "")
        .replace("ai_", "")
        .replace("0419_", "")
        .replace("0424_", "")
        .replace("0508_", "")
        .replace("_", " ")
        .title()
    )


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = RESULTS_DIR / "master_cross_generator_summary.csv"
    df = pd.read_csv(summary_path)

    df["accuracy_percent"] = df["accuracy"] * 100
    df["fake_recall_percent"] = df["fake_recall"] * 100
    df["model_display"] = df["model"].replace({
        "resnet18": "ResNet18",
        "efficientnet_b0": "EfficientNet-B0",
        "convnext_tiny": "ConvNeXt-Tiny"
    })
    df["test_generator_display"] = df["test_generator"].apply(clean_name)

    # Table 1: same-generator results only
    same_generator = df[df["test_generator"] == df["train_generator"]]
    same_generator_table = same_generator[[
        "model_display",
        "test_generator_display",
        "accuracy_percent",
        "fake_recall_percent",
        "fake_f1"
    ]].sort_values("accuracy_percent", ascending=False)

    same_generator_table.to_csv(
        RESULTS_DIR / "same_generator_model_comparison.csv",
        index=False
    )

    # Table 2: average cross-generator results excluding same generator
    cross_df = df[df["test_generator"] != df["train_generator"]]

    avg_cross = cross_df.groupby("model_display").agg({
        "accuracy_percent": "mean",
        "fake_recall_percent": "mean",
        "fake_f1": "mean"
    }).reset_index()

    avg_cross = avg_cross.sort_values("accuracy_percent", ascending=False)

    avg_cross.to_csv(
        RESULTS_DIR / "average_cross_generator_model_comparison.csv",
        index=False
    )

    print("=" * 70)
    print("Same-generator comparison")
    print("=" * 70)
    print(same_generator_table)

    print("\n" + "=" * 70)
    print("Average cross-generator comparison")
    print("=" * 70)
    print(avg_cross)

    # Chart 1: same-generator accuracy
    plt.figure(figsize=(8, 5))
    plt.bar(
        same_generator_table["model_display"],
        same_generator_table["accuracy_percent"]
    )
    plt.title("Same-Generator Accuracy on Midjourney")
    plt.xlabel("Model")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "same_generator_accuracy_comparison.png", dpi=300)
    plt.close()

    # Chart 2: average cross-generator accuracy
    plt.figure(figsize=(8, 5))
    plt.bar(
        avg_cross["model_display"],
        avg_cross["accuracy_percent"]
    )
    plt.title("Average Cross-Generator Accuracy")
    plt.xlabel("Model")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "average_cross_generator_accuracy.png", dpi=300)
    plt.close()

    # Chart 3: accuracy by generator and model
    pivot = df.pivot_table(
        index="test_generator_display",
        columns="model_display",
        values="accuracy_percent"
    )

    pivot.plot(kind="bar", figsize=(12, 6))
    plt.title("Cross-Generator Accuracy by Model")
    plt.xlabel("Test Generator")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cross_generator_accuracy_by_model.png", dpi=300)
    plt.close()

    print("\nSaved charts to figures folder.")


if __name__ == "__main__":
    main()