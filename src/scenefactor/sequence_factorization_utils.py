from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm

from src.scenefactor.data.common import NumpyTensor
from src.scenefactor.data.sequence import FrameSequence
from src.scenefactor.utils.geom import BBox, decompose_cmask, remove_artifacts, compute_bbox


SequenceBmasks = dict[int, dict[int, NumpyTensor['h', 'w']]] # label: index: bmask
SequenceBboxes = dict[int, dict[int, BBox]]                  # label: index: bbox


def compute_sequence_bmasks_bboxes(sequence: FrameSequence, instance2semantic: dict=None, min_area=1024) -> tuple[
    SequenceBmasks,
    SequenceBboxes
]:
    """
    TODO: parallelize
    """
    bmasks = defaultdict(dict)
    bboxes = defaultdict(dict)
    index = 0 # tqdm progress bar doesn't show with enumerate
    for imask in tqdm(sequence.imasks, desc='Sequence extraction computing bmasks and bboxes'):
        for label in np.unique(imask):
            if instance2semantic is not None:
                # additional filtering based on semantic info
                if label not in instance2semantic:
                    continue
                semantic_info = sequence.metadata['semantic_info'][instance2semantic[label]]
                if semantic_info['class'] == 'stuff':
                    continue
            bmask = imask == label
            bmask = remove_artifacts(bmask, mode='islands', min_area=min_area)
            bmask = remove_artifacts(bmask, mode='holes'  , min_area=min_area)
            if not np.any(bmask):
                continue
            bmasks[label][index] = bmask
            bboxes[label][index] = compute_bbox(bmask)
        index += 1
    return bmasks, bboxes


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