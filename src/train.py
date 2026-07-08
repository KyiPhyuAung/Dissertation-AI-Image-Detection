import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import DATASET_DIR, DEVICE, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS
from src.dataset import AIImageDataset
from src.transforms import get_train_transforms
from src.models import build_resnet18


def train():
    dataset = AIImageDataset(
        dataset_root=DATASET_DIR / "sample",
        transform=get_train_transforms()
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    model = build_resnet18().to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Using device: {DEVICE}")
    print(f"Training images: {len(dataset)}")

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

        print(
            f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
            f"Loss: {running_loss:.4f} "
            f"Accuracy: {accuracy:.2f}%"
        )


if __name__ == "__main__":
    train()