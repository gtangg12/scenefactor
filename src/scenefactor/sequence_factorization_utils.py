from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm

from src.scenefactor.data.common import NumpyTensor
from src.scenefactor.data.sequence import FrameSequence
from src.scenefactor.utils.geom import BBox, decompose_cmask, remove_artifacts, compute_bbox, connected_components


SequenceBmasks = dict[int, dict[int, NumpyTensor['h', 'w']]] # label: index: bmask
SequenceBboxes = dict[int, dict[int, BBox]]                  # label: index: bbox


def semantic_class(label: int, sequence: FrameSequence, instance2semantic: dict) -> str | None:
    """
    """
    if label not in instance2semantic:
        return None
    return sequence.metadata['semantic_info'][instance2semantic[label]]['class']


def compute_sequence_bmasks_bboxes(sequence: FrameSequence, instance2semantic: dict, min_area=1024, background=0) -> tuple[
    SequenceBmasks,
    SequenceBboxes
]:
    """
    TODO: parallelize
    """
    bmasks = defaultdict(dict)
    bboxes = defaultdict(dict)
    for index, imask in tqdm(enumerate(sequence.imasks), desc='Sequence extraction computing bmasks and bboxes'):
        for label in np.unique(imask):
            if semantic_class(label, sequence, instance2semantic) in [None, 'stuff']:
                continue
            bmask = imask == label
            bmask = remove_artifacts(bmask, mode='islands', min_area=min_area)
            bmask = remove_artifacts(bmask, mode='holes'  , min_area=min_area)
            if not np.any(bmask):
                continue
            bmasks[label][index] = bmask
            bboxes[label][index] = compute_bbox(bmask)
    return bmasks, bboxes


def dialate_bmask(bmask: NumpyTensor['h', 'w'], radius) -> NumpyTensor['h', 'w']:
    """
    """
    bmask = bmask.astype(np.uint8)
    bmask = cv2.dilate(bmask, np.ones((radius, radius)), iterations=1)
    return bmask.astype(bool)


def remove_background(
    image: NumpyTensor['h', 'w', 3], 
    bmask: NumpyTensor['h', 'w'], background=255, outline_thickness=1
) -> NumpyTensor['h', 'w', 3]:
    """
    """
    image = image.copy()
    image[~bmask] = background
    if outline_thickness:
        contours, _ = cv2.findContours(bmask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (0, 0, 0), outline_thickness)
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


def fill_background_holes(cmask: NumpyTensor['h', 'w'], max_area=4096, background=0) -> NumpyTensor['h', 'w']:
    """
    """
    cmask_filled = np.array(cmask)
    for label in np.unique(cmask):
        if label == background:
            continue
        for hole, area in compute_holes(cmask == label):
            if area > max_area:
                continue
            hole_labels, hole_counts = np.unique(cmask[hole], return_counts=True)
            if hole_labels[np.argmax(hole_counts)] == background:
                cmask_filled[hole] = label
    return cmask_filled