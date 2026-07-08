import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import (
    DEVICE,
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    CHECKPOINT_DIR,
    TINY_GENIMAGE_DIR,
)
from src.genimage_dataset import GenImageDataset
from src.transforms import get_train_transforms
from src.models import build_model


def train(model_name: str, generator: str):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = GenImageDataset(
        root_dir=TINY_GENIMAGE_DIR,
        generator=generator,
        split="train",
        transform=get_train_transforms(),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model = build_model(model_name).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_accuracy = 0.0

    print("=" * 70)
    print("Training Experiment")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Model: {model_name}")
    print(f"Training generator: {generator}")
    print(f"Training images: {len(dataset)}")
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

        accuracy = 100 * correct / total
        avg_loss = running_loss / len(dataloader)

        print(
            f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
            f"Loss: {avg_loss:.4f} "
            f"Accuracy: {accuracy:.2f}%"
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            checkpoint_path = CHECKPOINT_DIR / f"{model_name}_{generator}_best.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--generator", required=True)
    args = parser.parse_args()

    train(
        model_name=args.model,
        generator=args.generator,
    )