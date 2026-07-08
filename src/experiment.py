import argparse
import json
from datetime import datetime

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from torch.utils.data import DataLoader

from src.config import (
    DEVICE,
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    CHECKPOINT_DIR,
    RESULTS_DIR,
    TINY_GENIMAGE_DIR,
)
from src.genimage_dataset import GenImageDataset
from src.models import build_model
from src.transforms import get_train_transforms, get_eval_transforms


TEST_GENERATORS = [
    "imagenet_midjourney",
    "imagenet_ai_0424_sdv5",
    "imagenet_glide",
    "imagenet_ai_0419_biggan",
    "imagenet_ai_0508_adm",
    "imagenet_ai_0424_wukong",
    "imagenet_ai_0419_vqdm",
]


def train_model(model_name, train_generator, experiment_dir):
    dataset = GenImageDataset(
        root_dir=TINY_GENIMAGE_DIR,
        generator=train_generator,
        split="train",
        transform=get_train_transforms(),
    )

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = build_model(model_name).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_accuracy = 0.0
    checkpoint_path = CHECKPOINT_DIR / f"{model_name}_{train_generator}_best.pth"

    history = []

    print("=" * 70)
    print("Training")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Training generator: {train_generator}")
    print(f"Training images: {len(dataset)}")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    for epoch in range(NUM_EPOCHS):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in dataloader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = correct / total
        avg_loss = running_loss / len(dataloader)

        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "train_accuracy": accuracy,
        })

        print(
            f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
            f"Loss: {avg_loss:.4f} "
            f"Accuracy: {accuracy * 100:.2f}%"
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")

    history_path = experiment_dir / "training_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)

    return checkpoint_path


def evaluate_on_generator(model, generator):
    dataset = GenImageDataset(
        root_dir=TINY_GENIMAGE_DIR,
        generator=generator,
        split="val",
        transform=get_eval_transforms(),
    )

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

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
        labels=[0, 1],
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return {
        "test_generator": generator,
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


def run_experiment(model_name, train_generator):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{timestamp}_{model_name}_train_{train_generator}"

    experiment_dir = RESULTS_DIR / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    config = {
        "model": model_name,
        "train_generator": train_generator,
        "device": str(DEVICE),
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
        "dataset": "tiny-genimage",
    }

    with open(experiment_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)

    checkpoint_path = train_model(
        model_name=model_name,
        train_generator=train_generator,
        experiment_dir=experiment_dir,
    )

    model = build_model(model_name).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))

    results = []

    print("=" * 70)
    print("Cross-Generator Evaluation")
    print("=" * 70)

    for test_generator in TEST_GENERATORS:
        metrics = evaluate_on_generator(model, test_generator)
        metrics["train_generator"] = train_generator
        metrics["model"] = model_name
        results.append(metrics)

        print(
            f"{test_generator}: "
            f"Accuracy {metrics['accuracy'] * 100:.2f}% | "
            f"Fake Recall {metrics['fake_recall'] * 100:.2f}% | "
            f"Fake F1 {metrics['fake_f1']:.4f}"
        )

    results_path = experiment_dir / "cross_generator_results.csv"
    pd.DataFrame(results).to_csv(results_path, index=False)

    print("=" * 70)
    print("Experiment completed.")
    print(f"Saved results to: {experiment_dir}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-generator", required=True)

    args = parser.parse_args()

    run_experiment(
        model_name=args.model,
        train_generator=args.train_generator,
    )