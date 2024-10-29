import numpy as np
import torch
from PIL import Image

from scenefactor.data.common import NumpyTensor, TorchTensor


Tensor = NumpyTensor | TorchTensor


def read_image(filename: str) -> NumpyTensor['h', 'w', 'channels']:
    """
    Read image from file and return as numpy array.
    """
    return np.array(Image.open(filename))


def tensor_to_image(tensor: TorchTensor['batch', 'channels', 'h', 'w']) -> list[NumpyTensor['h', 'w', 'channels']]:
    """
    Convert tensor to image.
    """
    tensor = tensor.detach().cpu().numpy()
    tensor = tensor.transpose(0, 2, 3, 1)
    tensor = np.clip(tensor * 255, 0, 255).astype(np.uint8)
    return list(tensor)


def image_to_tensor(image: NumpyTensor['...', 'h', 'w', 'channels']) -> TorchTensor['B', 'channels', 'h', 'w']:
    """
    Convert image to tensor.
    """
    image = image.transpose(0, 3, 1, 2) if image.ndim == 4 else image.transpose(2, 0, 1)[None, ...]
    image = image / 255
    return torch.from_numpy(image).float()


def untile(grid: Tensor['h', 'w', 'channels'], H: int, W: int) -> Tensor['n', 'h // H', 'w // W', 'channels']:
    """
    """
    h, w = grid.shape[:2]
    assert h % H == 0 and w % W == 0, f'grid shape {grid.shape} not divisible by {H}x{W}'
    h_tile = h // H
    w_tile = w // W
    tiles = [
        grid[
            rindex * h_tile:(rindex + 1) * h_tile, 
            cindex * w_tile:(cindex + 1) * w_tile, ...
        ]
        for rindex in range(H) 
        for cindex in range(W)
    ]
    return np.array(tiles) if isinstance(grid, np.ndarray) else torch.stack(tiles)