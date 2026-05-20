import os

from PIL import Image
from torchvision import datasets, transforms


DEFAULT_IMAGE_EXTENSIONS = (
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
)


def parse_channel_values(value, default):
    """Parse comma separated channel values used for normalization."""
    if value is None or value == "":
        return default
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if len(values) == 1:
        return values * 3
    if len(values) != 3:
        raise ValueError("Expected one value or three comma separated channel values.")
    return values


def list_image_files(root_dir, extensions=None):
    extensions = tuple(ext.lower() for ext in (extensions or DEFAULT_IMAGE_EXTENSIONS))
    image_paths = []
    for current_root, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(extensions):
                image_paths.append(os.path.join(current_root, filename))
    image_paths.sort()
    return image_paths


def build_image_transform(image_size, mean=None, std=None, center_crop=False):
    transform_steps = [transforms.Resize((image_size, image_size))]
    if center_crop:
        transform_steps.append(transforms.CenterCrop(image_size))
    transform_steps.append(transforms.ToTensor())
    if mean is not None and std is not None:
        transform_steps.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(transform_steps)


class UnlabeledMedicalImageDataset(object):
    """Recursively loads unlabeled scattergram/histogram images for pretraining."""

    def __init__(self, root_dir, image_size=224, transform=None, extensions=None):
        self.root_dir = root_dir
        self.image_paths = list_image_files(root_dir, extensions=extensions)
        if not self.image_paths:
            raise RuntimeError("No image files found under: {}".format(root_dir))
        self.transform = transform or build_image_transform(image_size=image_size)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path = self.image_paths[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            image = self.transform(image)
        return image, path


def build_unlabeled_dataset(root_dir, image_size, mean=None, std=None):
    transform = build_image_transform(image_size=image_size, mean=mean, std=std)
    return UnlabeledMedicalImageDataset(root_dir=root_dir, image_size=image_size, transform=transform)


def build_classification_dataset(root_dir, image_size, mean=None, std=None):
    transform = build_image_transform(image_size=image_size, mean=mean, std=std)
    return datasets.ImageFolder(root=root_dir, transform=transform)
