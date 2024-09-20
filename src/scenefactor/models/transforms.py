from typing import Tuple

import torch
import torch.nn as nn
from torchvision import transforms


def transform_imagenet(resize: Tuple[int, int] = (224, 224), interpolation=transforms.InterpolationMode.BICUBIC):
    """
    Returns transform that resizes and normalizes an image with ImageNet mean and std.
    """
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(resize, interpolation=interpolation),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225],
            ),
        ]
    )