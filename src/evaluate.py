import torch
from torch.utils.data import DataLoader

from src.config import DATASET_DIR, DEVICE, BATCH_SIZE, CHECKPOINT_DIR
from src.dataset import AIImageDataset
from src.transforms import get_eval_transforms
from src.models import build_resnet18


def evaluate():
    dataset = AIImageDataset(
        dataset_root=DATASET_DIR / "sample",
        transform=get_eval_transforms()
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = build_resnet18().to(DEVICE)

    checkpoint_path = CHECKPOINT_DIR / "resnet18_sample_best.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))

    model.eval()

    correct = 0
    total = 0

    class_names = ["real", "fake"]

    print(f"Using device: {DEVICE}")
    print(f"Evaluation images: {len(dataset)}")
    print("=" * 50)

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for true_label, pred_label, probs in zip(labels, predicted, probabilities):
                print(
                    f"True: {class_names[true_label.item()]} | "
                    f"Predicted: {class_names[pred_label.item()]} | "
                    f"Confidence: {probs[pred_label].item():.4f}"
                )

    accuracy = 100 * correct / total

    print("=" * 50)
    print(f"Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    evaluate()