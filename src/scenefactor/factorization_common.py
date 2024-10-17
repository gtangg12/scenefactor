import pickle
import os
import shutil
from pathlib import Path

import cv2
import numpy as np


from src.scenefactor.data.common import NumpyTensor
from src.scenefactor.data.sequence import FrameSequence
from src.scenefactor.utils.geom import dialate_bmask


SEMANTIC_BACKGROUND = 0
INSTANCE_BACKGROUND = 0


def semantic_class(label: int, sequence: FrameSequence, instance2semantic: dict) -> str | None:
    """
    """
    if label not in instance2semantic:
        return None
    return sequence.metadata['semantic_info'][instance2semantic[label]]['class']


def compute_inpaint_radius(
    bmask: NumpyTensor['h', 'w'], 
    imask: NumpyTensor['h', 'w'], 
    ratio: float, clip_min=15, clip_max=75, bound_iterations=20, bound_threshold=5
) -> int:
    """
    """
    # assume bmask is a circle: r = sqrt(VOL / pi)(sqrt(c) - 1)
    assert ratio > 1
    radius = int(np.sqrt(np.sum(bmask) / np.pi * (ratio - 1)))
    radius = int(np.clip(radius, clip_min, clip_max))

    # ensure dialation doesn't overpaint non adjacent regions
    num_labels = len(np.unique(imask[dialate_bmask(bmask, clip_min)]))
    lo = 0
    hi = radius
    for _ in range(bound_iterations):
        mid = (lo + hi) // 2
        num_labels_current = len(np.unique(imask[dialate_bmask(bmask, mid)]))
        if num_labels_current > num_labels:
            hi = mid
        else:
            lo = mid
        if hi - lo < bound_threshold:
            break
    radius = min(radius, lo)
    return radius