import numpy as np
import torch
from omegaconf import OmegaConf
from trimesh.base import Trimesh

from scenefactor.data.common import TorchTensor
from src.scenefactor.data.sequence import FrameSequence
from scenefactor.models import ImageTo3DModel, ModelStableDiffusion


class SequenceExtraction:
    """
    """
    def __init__(self, config: OmegaConf):
        """
        """
        self.config = config
    
    def __call__(self, sequence: FrameSequence) -> dict[int, Trimesh]:
        """
        """
        imask_labels = np.unique(sequence.imask)
        for label in imask_labels:
            pass
            # determine if each label is a good view using sequence by looking at holes in instance mask
            # for each good view, select frame with highest score, and ensure everything is above threshold (try with clip, or grounding dino)
        # image to 3d model for all good views
        # return dict mapping label to mesh
        pass


class SequenceInpainting:
    """
    """
    def __init__(self, config: OmegaConf):
        """
        """
        self.config = config

    def __call__(self, sequence: FrameSequence, labels: set[int]) -> FrameSequence:
        """
        """
        sequence = sequence.clone()
        # get labels from sequence metadata, labels passed as arg is to be inpainted
        # for label in labels
        # for each frame with label, remove the label and inpaint with stable diffusion
        # **no need for propagation since image 2 3d selected only using one frame**
        # update sequence with inpaintings
        pass


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
    
