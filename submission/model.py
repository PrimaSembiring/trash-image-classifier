import timm
from torchvision import transforms
from torchvision.transforms import InterpolationMode

MODEL_NAME = "vit_small_patch14_dinov2.lvd142m"
NUM_CLASSES = 5
IMAGE_SIZE = 224

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# urutan kelas = indeks label 0..4
CLASS_NAMES = [
    "botol_plastik",
    "gelas_plastik",
    "kertas_kardus",
    "saset_kemasan",
    "styrofoam",
]


def get_model():
    # pretrained=False: bobot dimuat dari model.pth, tidak ada unduhan
    return timm.create_model(
        MODEL_NAME,
        pretrained=False,
        num_classes=NUM_CLASSES,
        img_size=IMAGE_SIZE,
    )


def get_transform():
    # deterministik, sama dengan transformasi validation saat training
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])