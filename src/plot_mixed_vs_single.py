import pandas as pd
import matplotlib.pyplot as plt

from src.config import RESULTS_DIR, FIGURES_DIR


MODEL_NAMES = {
    "resnet18": "ResNet18",
    "efficientnet_b0": "EfficientNet-B0",
    "convnext_tiny": "ConvNeXt-Tiny",
}

COMMON_UNSEEN_GENERATORS = [
    "imagenet_ai_0419_biggan",
    "imagenet_ai_0508_adm",
    "imagenet_ai_0424_wukong",
    "imagenet_ai_0419_vqdm",
]


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RESULTS_DIR / "master_cross_generator_summary.csv")

    single = df[
        (df["train_generator"] == "imagenet_midjourney")
        & (df["test_generator"].isin(COMMON_UNSEEN_GENERATORS))
    ]

    mixed = df[
        (df["train_generator"] == "mixed")
        & (df["test_generator"].isin(COMMON_UNSEEN_GENERATORS))
    ]

    single_avg = (
        single.groupby("model")["accuracy"]
        .mean()
        .reset_index()
        .rename(columns={"accuracy": "single_training_accuracy"})
    )

    mixed_avg = (
        mixed.groupby("model")["accuracy"]
        .mean()
        .reset_index()
        .rename(columns={"accuracy": "mixed_training_accuracy"})
    )

    comparison = pd.merge(single_avg, mixed_avg, on="model")
    comparison["improvement"] = (
        comparison["mixed_training_accuracy"]
        - comparison["single_training_accuracy"]
    )

    comparison["model_display"] = comparison["model"].map(MODEL_NAMES)

    for col in [
        "single_training_accuracy",
        "mixed_training_accuracy",
        "improvement",
    ]:
        comparison[col] = comparison[col] * 100

    comparison = comparison.sort_values(
        "mixed_training_accuracy",
        ascending=False
    )

    output_csv = RESULTS_DIR / "mixed_vs_single_common_unseen_comparison.csv"
    comparison.to_csv(output_csv, index=False)

    print(comparison[[
        "model_display",
        "single_training_accuracy",
        "mixed_training_accuracy",
        "improvement",
    ]])

    x = range(len(comparison))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(
        [i - width / 2 for i in x],
        comparison["single_training_accuracy"],
        width,
        label="Single-generator training",
    )
    plt.bar(
        [i + width / 2 for i in x],
        comparison["mixed_training_accuracy"],
        width,
        label="Mixed-generator training",
    )

    plt.xticks(x, comparison["model_display"])
    plt.ylabel("Average Accuracy (%)")
    plt.xlabel("Model")
    plt.title("Single vs Mixed Training on Common Unseen Generators")
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()

    output_fig = FIGURES_DIR / "mixed_vs_single_common_unseen_generators.png"
    plt.savefig(output_fig, dpi=300)
    plt.close()

    print("Saved table to:", output_csv)
    print("Saved figure to:", output_fig)


if __name__ == "__main__":
    main()