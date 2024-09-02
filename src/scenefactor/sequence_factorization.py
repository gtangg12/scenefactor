import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from trimesh.base import Trimesh

from scenefactor.data.common import NumpyTensor
from scenefactor.data.sequence import FrameSequence
from scenefactor.models import ModelInstantMesh, ModelStableDiffusion
from scenefactor.utils.geom import bmask_iou


def compute_bbox(bmask: NumpyTensor['h', 'w']) -> tuple[int, int, int, int]:
    """
    Returns the bounding box of a binary mask, where the minimum is inclusive and the maximum is exclusive.
    """
    rows = np.any(bmask, axis=1)
    cols = np.any(bmask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmax += 1
    cmax += 1
    return rmin, rmax, cmin, cmax


def compute_holes(bmask: NumpyTensor['h', 'w']) -> list[tuple[NumpyTensor['h', 'w'], int]]:
    """
    Computes the holes in a binary mask. Assumes (0, 0) is background (can check with compute_bbox).
    """
    bmask = ~bmask
    nregions, regions, stats, _ = cv2.connectedComponentsWithStats(bmask, 8)
    holes = []
    for i in range(1, nregions): # index 0 refers to all pixels with label 0 (object)
        if regions[i][0, 0]:
            continue
        holes.append((regions[i], stats[i, cv2.CC_STAT_AREA])) # (region bmask, area)
    return holes


class SequenceExtraction:
    """
    """
    def __init__(self, config: OmegaConf):
        """
        """
        self.config = config
        self.model = ModelInstantMesh(config.model)
    
    def __call__(self, sequence: FrameSequence) -> dict[int, Trimesh]:
        """
        """
        def compute_view_score(bmask: NumpyTensor['h', 'w']) -> bool:
            """
            """
            # make sure you can get a good crop (return 0 if not)
            # determine if each label is a good view using sequence by looking at holes in instance mask as well as ensuring object is not close to edges
            pass

        def process(label: int, score_threshold=0.5) -> Trimesh | None:
            """
            """
            max_score, max_view = 0, None
            for i in range(len(sequence)):
                label_bmask = sequence[i].imask == label
                score, view = compute_view_score(label_bmask)
                if score > score_threshold and score > max_score:
                    max_score, max_view = score, view
            if max_view is None:
                return None
            return self.model.process(max_view)

        meshes = {}
        for label in np.unique(sequence.imask):
            mesh = process(label)
            if mesh is not None:
                meshes[label] = mesh
        return meshes


class SequenceInpainting:
    """
    """
    INPAINT_PROMPT = 'Fill in the missing parts of the image.'

    def __init__(self, config: OmegaConf):
        """
        """
        self.config = config
        self.model_inpainter = ModelStableDiffusion(config.model_inpainter)
        self.model_segmenter = ModelSAM2           (config.model.segmenter)

    def __call__(self, sequence: FrameSequence, labels_to_inpaint: set[int]) -> FrameSequence:
        """

        NOTE:: we do not propagate inpainting e.g. using NeRF since image to 3d involves only one frame
        """ 
        def predict_label(index: int, bmask: NumpyTensor['h', 'w']) -> int:
            """
            """
            frame = sequence[index]
            return np.argmax(np.bincount(frame.imask[bmask]))

        def inpaint(index: int, label: int):
            """
            """
            frame = sequence[index]
            bmask = frame.imask == label
            frame.image = self.model_inpainter(self.INPAINT_PROMPT, frame.image, bmask)
            for bmask_inpainted in self.model_segmenter(frame.image):
                if bmask_iou(bmask, bmask_inpainted) > self.config.bmask_iou_threshold:
                    # inpainting operates on tensor views of sequence
                    frame.imask[bmask & bmask_inpainted] = predict_label(index, bmask_inpainted)

        sequence = sequence.clone()
        for label in labels_to_inpaint:
            for i in range(len(sequence)):
                inpaint(i, label)
        return sequence


class SequenceFactorization:
    """
    """
    def __init__(self, config: OmegaConf):
        """
        """
        self.config = config
        self.extractor = SequenceExtraction(config.extractor)
        self.inpainter = SequenceInpainting(config.inpaintor)

    def __call__(self, sequence: FrameSequence) -> dict[int, Trimesh]:
        """
        """
        sequence = sequence.clone()
        sequence.metadata['labels'] = set(np.unique(sequence.imask))
        meshes_total = {}
        while True: # either use num iters or until no more new labels extracted
            meshes = self.extractor(sequence)
            sequence = self.inpainter(sequence, meshes.keys())
            meshes_total.update(meshes)
        return meshes_total


'''
Ideas for metrics:

replica (scannet?): look at semantics and get for each scene all the non background objects
    proportion of non background objects extracted to all non background objects
    proportion of background objects extracted to all background objects
    ratio of non background objects to background objects extracted

clip scores of rendered objects vs their gt objects vs images to measure extraction quality

pose errors in registration process

qualitative results of meshes
qualitative results of inpainting
qualitative results of editing/removal in context of whole scene
'''