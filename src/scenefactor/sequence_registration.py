import torch
from trimesh.base import Trimesh


from scenefactor.data.common import TorchTensor
from src.scenefactor.data.sequence import FrameSequenceDataset
from scenefactor.renderer.renderer import Renderer


class SequenceRegistration:
    """
    """
    def __init__(self, sequence: FrameSequenceDataset):
        """
        """
        self.sequence = sequence

    def __call__(self, mesh: Trimesh) -> tuple[TorchTensor[4, 4], float]:
        """
        """
        # renderer sequence: render from n views in regular polyhedra format
        # compute all pairwise loftr matchings
        # do m to n max biparate matching, with edge cost as # of loftr matches
        # compute 3d positions of matched pixels in sequence
        # compute 3d positions of matched pixels in rendered sequence
        # use 6dof + scale optimization: https://gist.github.com/nh2/bc4e2981b0e213fefd4aaa33edfb3893
        # returns 6dof pose + scale
        pass
