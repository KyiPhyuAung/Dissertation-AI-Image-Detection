import torch.nn as nn
from torchvision import models

from src.config import NUM_CLASSES


def build_model(model_name: str):

    model_name = model_name.lower()

    if model_name == "resnet18":

        model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, NUM_CLASSES)

        return model

    elif model_name == "efficientnet_b0":

        model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )

        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(
            in_features,
            NUM_CLASSES
        )

        return model

    elif model_name == "convnext_tiny":

        model = models.convnext_tiny(
            weights=models.ConvNeXt_Tiny_Weights.DEFAULT
        )

        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(
            in_features,
            NUM_CLASSES
        )

        return model

    else:
        raise ValueError(f"Unknown model: {model_name}")