import torch
from trimesh.base import Trimesh
from omegaconf import OmegaConf

from scenefactor.data.common import NumpyTensor
from scenefactor.data.sequence import FrameSequence
from scenefactor.renderer.renderer import Renderer


class SequenceRegistration:
    """
    """
    def __init__(self, config: OmegaConf):
        """
        """
        self.config = config

    def __call__(self, mesh: Trimesh, label: int, sequence: FrameSequence) -> tuple[NumpyTensor[4, 4], float]:
        """
        """
        # compute pose for object denoted by label
        # renderer sequence: render from n views in regular polyhedra format (see renderer and camera_generation)
        # compute all pairwise loftr matchings
        # do m to n max biparate matching, with edge cost as # of loftr matches
        # compute 3d positions of matched pixels in sequence using pose in frame sequence + gt depths
        # compute 3d positions of matched pixels in rendered sequence using same tactic (renderer provides gt depths)
        # use 6dof + scale optimization: https://gist.github.com/nh2/bc4e2981b0e213fefd4aaa33edfb3893
        # returns 6dof pose + scale
        pass