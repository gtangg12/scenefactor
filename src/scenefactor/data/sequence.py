import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from src.scenefactor.data.sequence_reader import FrameSequenceReader


@dataclass
class FrameSequence:
    """
    """
    pose: NumpyTensor['n', 4, 4]

    image: NumpyTensor['n', 'h', 'w']
    depth: NumpyTensor['n', 'h', 'w'] = None
    smask: NumpyTensor['n', 'h', 'w'] = None
    imask: NumpyTensor['n', 'h', 'w'] = None
    bmask: NumpyTensor['n', 'h', 'w'] = None

    metadata: dict = field(default_factory=dict)

    def __len__(self):
        """
        """
        return len(self.image)

    def __getitem__(self, index: int):
        """
        """
        return FrameSequence(**{k: tensor[index] for k, tensor in asdict(self).items()})

    def __slice__(self, index1: int, index2: int):
        """
        """
        return FrameSequence(**{k: tensor[index1:index2] for k, tensor in asdict(self).items()})
    
    def clone(self):
        """
        """
        return deepcopy.copy(self)