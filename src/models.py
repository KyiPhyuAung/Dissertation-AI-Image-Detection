import torch.nn as nn
from torchvision import models
from src.config import NUM_CLASSES


def build_resnet18():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, NUM_CLASSES)

    return model