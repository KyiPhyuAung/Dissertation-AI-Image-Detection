import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from src.config import DEVICE, BATCH_SIZE, CHECKPOINT_DIR, TINY_GENIMAGE_DIR
from src.genimage_dataset import GenImageDataset
from src.transforms import get_eval_transforms
from src.models import build_model


MODEL_NAME = "efficientnet_b0"
GENERATOR = "imagenet_ai_0424_sdv5"
SPLIT = "val"


def evaluate():
    dataset = GenImageDataset(
        root_dir=TINY_GENIMAGE_DIR,
        generator=GENERATOR,
        split=SPLIT,
        transform=get_eval_transforms()
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = build_model(MODEL_NAME).to(DEVICE)

    checkpoint_path = CHECKPOINT_DIR / f"{MODEL_NAME}_imagenet_midjourney_best.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()

    y_true = []
    y_pred = []

    print(f"Using device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")
    print(f"Generator: {GENERATOR}")
    print(f"Split: {SPLIT}")
    print(f"Evaluation images: {len(dataset)}")
    print("=" * 50)

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(DEVICE)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print("Accuracy:", f"{accuracy * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["real", "fake"],
            digits=4
        )
    )


if __name__ == "__main__":
    evaluate()