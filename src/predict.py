import argparse
from pathlib import Path

import torch
from PIL import Image

from src.config import DEVICE, CHECKPOINT_DIR
from src.transforms import get_eval_transforms
from src.models import build_resnet18


def predict_image(image_path):
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    class_names = ["real", "fake"]

    transform = get_eval_transforms()
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    model = build_resnet18().to(DEVICE)
    checkpoint_path = CHECKPOINT_DIR / "resnet18_sample_best.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    predicted_class = class_names[predicted.item()]

    print("=" * 50)
    print(f"Image: {image_path}")
    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {confidence.item():.4f}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to image file")
    args = parser.parse_args()

    predict_image(args.image)