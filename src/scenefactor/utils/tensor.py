import numpy as np
import torch

from scenefactor.data.common import NumpyTensor, TorchTensor


def tensor_to_image(tensor: TorchTensor['batch', 'channels', 'h', 'w']) -> list[NumpyTensor['h', 'w', 'channels']]:
    """
    Convert tensor to image.
    """
    tensor = tensor.detach().cpu().numpy()
    tensor = tensor.transpose(0, 2, 3, 1)
    tensor = np.clip(tensor * 255, 0, 255).astype(np.uint8)
    return list(tensor)