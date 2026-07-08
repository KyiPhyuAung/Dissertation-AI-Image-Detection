import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from src.config import DEVICE, BATCH_SIZE, CHECKPOINT_DIR, TINY_GENIMAGE_DIR, RESULTS_DIR
from src.genimage_dataset import GenImageDataset
from src.transforms import get_eval_transforms
from src.models import build_model


MODEL_NAME = "efficientnet_b0"
TRAIN_GENERATOR = "imagenet_midjourney"
SPLIT = "val"

TEST_GENERATORS = [
    "imagenet_midjourney",
    "imagenet_ai_0424_sdv5",
    "imagenet_glide",
    "imagenet_ai_0419_biggan",
    "imagenet_ai_0508_adm",
    "imagenet_ai_0424_wukong",
    "imagenet_ai_0419_vqdm",
]


def evaluate_generator(model, generator_name):
    dataset = GenImageDataset(
        root_dir=TINY_GENIMAGE_DIR,
        generator=generator_name,
        split=SPLIT,
        transform=get_eval_transforms()
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    y_true = []
    y_pred = []

    model.eval()

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(DEVICE)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=None,
        labels=[0, 1]
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return {
        "train_generator": TRAIN_GENERATOR,
        "test_generator": generator_name,
        "split": SPLIT,
        "num_images": len(dataset),
        "accuracy": accuracy,
        "real_precision": precision[0],
        "real_recall": recall[0],
        "real_f1": f1[0],
        "fake_precision": precision[1],
        "fake_recall": recall[1],
        "fake_f1": f1[1],
        "tn": cm[0, 0],
        "fp": cm[0, 1],
        "fn": cm[1, 0],
        "tp": cm[1, 1],
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Cross-Generator Evaluation")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Trained on: {TRAIN_GENERATOR}")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    model = build_model(MODEL_NAME).to(DEVICE)

    checkpoint_path = CHECKPOINT_DIR / f"{MODEL_NAME}_{TRAIN_GENERATOR}_best.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))

    results = []

    for generator in TEST_GENERATORS:
        print(f"Evaluating on: {generator}")

        metrics = evaluate_generator(model, generator)
        results.append(metrics)

        print(
            f"Accuracy: {metrics['accuracy'] * 100:.2f}% | "
            f"Fake Recall: {metrics['fake_recall'] * 100:.2f}% | "
            f"Fake F1: {metrics['fake_f1']:.4f}"
        )
        print("-" * 70)

    df = pd.DataFrame(results)

    output_path = RESULTS_DIR / f"{MODEL_NAME}_trained_on_{TRAIN_GENERATOR}_cross_generator_results.csv"
    df.to_csv(output_path, index=False)

    print("=" * 70)
    print("Saved results to:")
    print(output_path)
    print("=" * 70)


if __name__ == "__main__":
    main()