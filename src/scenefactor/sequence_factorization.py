import numpy as np
import torch
from omegaconf import OmegaConf
from trimesh.base import Trimesh

from scenefactor.data.common import TorchTensor
from src.scenefactor.data.sequence import FrameSequence
from scenefactor.models import ImageTo3DModel, ModelStableDiffusion


class SequenceSegmentation:
    """
    """
    def __init__(self, config: OmegaConf, sequence: FrameSequence):
        """
        """
        pass

    def __call__(self):
        """
        """
        pass


class SequenceObjectExtraction:
    """
    """
    def __init__(self, config: OmegaConf):
        """
        """
        pass
    
    def __call__(self, sequence: FrameSequence) -> dict[int, Trimesh]:
        """
        """
        # get labels from sequence metadata
        # for label in labels (labels that need to be extracted)
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
        pass

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
        pass

    def __call__(self, sequence: FrameSequence) -> dict[int, Trimesh]:
        """
        """
        sequence = sequence.clone()
        sequence.metadata['labels'] = set(np.unique(sequence.imask))
        meshes_total = {}
        while True: # either use num iters or until all labels used
            meshes = self.sequence_object_extraction(sequence)
            sequence = self.sequence_inpainting(sequence, meshes.keys())
            meshes_total.update(meshes)
        return meshes_total