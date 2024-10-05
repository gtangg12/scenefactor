import cv2
import numpy as np
from PIL import Image

from scenefactor.data.common import NumpyTensor
from scenefactor.utils.geom import BBox, combine_bmasks


BLACK = (  0,   0,   0)
WHITE = (255, 255, 255)
RED   = (255,   0,   0)
GREEN = (  0, 255,   0)
BLUE  = (  0,   0, 255)


def colormap_image(image: NumpyTensor['h', 'w', 3]) -> Image.Image:
    """
    """
    return Image.fromarray(image.astype(np.uint8))


def colormap_tiles(tiles: list[NumpyTensor['h', 'w', 3]], r: int, c: int) -> Image.Image:
    """
    """
    h, w = tiles[0].shape[:2]
    assert all(tile.shape[:2] == (h, w) for tile in tiles), 'tiles must have the same shape'
    image = np.concatenate([
        np.concatenate(tiles[cindex * c:(cindex + 1) * c], axis=1)
        for cindex in range(r)
    ], axis=0)
    return Image.fromarray(image.astype(np.uint8))


def colormap_depth(depth: NumpyTensor['h', 'w']) -> Image.Image:
    """
    """
    depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
    depth = np.clip(depth, 0, 1)
    depth = np.stack([depth] * 3, axis=-1)
    return Image.fromarray((depth * 255).astype(np.uint8))


def colormap_cmask(
    cmask: NumpyTensor['h', 'w'], 
    image: NumpyTensor['h', 'w' ,3]=None, background=WHITE, foreground=None, blend=0.25
) -> Image.Image:
    """
    """
    # deterministic background color
    palette = np.random.RandomState(0).randint(0, 255, (np.max(cmask) + 1, 3))
    palette[0] = background
    if foreground is not None:
        for i in range(1, len(palette)):
            palette[i] = foreground
    image_mask = palette[cmask.astype(int)] # type conversion for boolean masks
    image_blend = image_mask if image is None else image_mask * (1 - blend) + image * blend
    image_blend = np.clip(image_blend, 0, 255).astype(np.uint8)
    return Image.fromarray(image_blend)


def colormap_bmask(bmask: NumpyTensor['h', 'w']) -> Image.Image:
    """
    """
    return colormap_cmask(bmask, background=BLACK, foreground=WHITE)


def colormap_bmasks(
    masks: NumpyTensor['n', 'h', 'w'], 
    image: NumpyTensor['h', 'w', 3]=None, background=WHITE, blend=0.25
) -> Image.Image:
    """
    """
    cmask = combine_bmasks(masks, sort=True)
    return colormap_cmask(cmask, image, background=background, blend=blend)


def colormap_bbox(bbox: BBox, image: NumpyTensor['h', 'w', 3], color=GREEN) -> Image.Image:
    """
    """
    image_bbox = image.copy()
    cv2.rectangle(image_bbox, (bbox[1], bbox[0]), (bbox[3], bbox[2]), color, 2)
    return Image.fromarray(image_bbox)


def colormap_bboxes(bboxes: list[BBox], image: NumpyTensor['h', 'w', 3], color=GREEN) -> Image.Image:
    """
    """
    image_bboxes = image.copy()
    for bbox in bboxes:
        cv2.rectangle(image_bboxes, (bbox[1], bbox[0]), (bbox[3], bbox[2]), color, 2)
    return Image.fromarray(image_bboxes)