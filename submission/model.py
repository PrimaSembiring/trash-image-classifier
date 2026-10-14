import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0

NUM_CLASSES = 5
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_model():
    # struktur EfficientNet-B0 dengan 5 output; bobot dimuat evaluator dari model.pth
    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
    return model


def get_transform():
    # transform inference (deterministik), sama dengan validation pipeline di notebook
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

# # Simpan bobot dengan:
# torch.save(model.state_dict(), "submission/model.pth")