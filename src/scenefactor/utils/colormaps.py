import numpy as np
from PIL import Image

from scenefactor.data.common import NumpyTensor
from scenefactor.utils.geom import combine_bmasks


def colormap_depth(depth: NumpyTensor['h', 'w']) -> Image.Image:
    """
    """
    depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
    depth = np.clip(depth, 0, 1)
    depth = np.stack([depth] * 3, axis=-1)
    return Image.fromarray((depth * 255).astype(np.uint8))


def colormap_mask(
    mask : NumpyTensor['h w'], 
    image: NumpyTensor['h w 3']=None, background=np.array([255, 255, 255]), foreground=None, blend=0.25
) -> Image.Image:
    """
    """
    palette = np.random.randint(0, 255, (np.max(mask) + 1, 3))
    palette[0] = background
    if foreground is not None:
        for i in range(1, len(palette)):
            palette[i] = foreground
    image_mask = palette[mask.astype(int)] # type conversion for boolean masks
    image_blend = image_mask if image is None else image_mask * (1 - blend) + image * blend
    image_blend = np.clip(image_blend, 0, 255).astype(np.uint8)
    return Image.fromarray(image_blend)


def colormap_bmask(bmask: NumpyTensor['h w']) -> Image.Image:
    """
    """
    return colormap_mask(bmask, background=np.array([0, 0, 0]), foreground=np.array([255, 255, 255]))


def colormap_bmasks(
    masks: NumpyTensor['n h w'], 
    image: NumpyTensor['h w 3']=None, background=np.array([255, 255, 255]), blend=0.25
) -> Image.Image:
    """
    """
    mask = combine_bmasks(masks)
    return colormap_mask(mask, image, background=background, blend=blend)