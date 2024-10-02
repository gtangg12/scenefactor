import torch
import torch.nn as nn
from torchvision import transforms


def transform_imagenet(resize: tuple[int, int]=None, interpolation=transforms.InterpolationMode.BICUBIC):
    """
    Returns transform that resizes and normalizes an image with ImageNet mean and std.
    """
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(resize, interpolation=interpolation) if resize else lambda x: x,
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225],
            ),
        ]
    )