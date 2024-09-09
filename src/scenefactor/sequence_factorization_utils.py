import cv2
import numpy as np

from src.scenefactor.data.common import NumpyTensor
from src.scenefactor.utils.geom import decompose_cmask


def remove_background(
    image: NumpyTensor['h', 'w', 3], 
    bmask: NumpyTensor['h', 'w'], 
    background=255
) -> NumpyTensor['h', 'w', 3]:
    """
    """
    image = image.copy()
    image[~bmask] = background
    return image


def compute_holes(bmask: NumpyTensor['h', 'w']) -> list[tuple[NumpyTensor['h', 'w'], int]]:
    """
    Computes the holes in a binary mask. Assumes (0, 0) is background (can check with compute_bbox).
    """
    bmask = ~bmask
    nregions, regions, stats, _ = cv2.connectedComponentsWithStats(bmask.astype(np.uint8), 8)
    region_bmasks = decompose_cmask(regions)
    holes = []
    for i in range(1, nregions): # index 0 refers to all pixels with label 0 (object)
        if region_bmasks[i][0, 0]: # TODO: don't assume background defined as having same value as (0, 0)
            continue
        holes.append((region_bmasks[i], stats[i, cv2.CC_STAT_AREA])) # (region bmask, area)
    return holes