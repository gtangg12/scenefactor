import copy
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from src.scenefactor.data.common import NumpyTensor
from src.scenefactor.data.sequence_reader import FrameSequenceReader


@dataclass
class FrameSequence:
    """
    """
    poses: NumpyTensor['n', 4, 4]
    
    images: NumpyTensor['n', 'h', 'w', 3]
    depths: NumpyTensor['n', 'h', 'w'] = None
    smasks: NumpyTensor['n', 'h', 'w'] = None
    imasks: NumpyTensor['n', 'h', 'w'] = None

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
        return copy.deepcopy(self)
    
    def save(self, filename: Path | str):
        """
        Saves data from a FrameSequence object to disk.
        """
        os.makedirs(filename, exist_ok=True)
        for k, v in asdict(self).items():
            if isinstance(v, np.ndarray):
                np.save(filename / f'{k}.npy', v)
        with open(filename / 'metadata.json', 'w') as f:
            json.dump(self.metadata, f)

    def load(self, filename: Path | str):
        """
        Loads data from disk into a FrameSequence object, overwriting existing data.
        """
        for k, v in asdict(self).items():
            if isinstance(v, np.ndarray):
                setattr(self, k, np.load(filename / f'{k}.npy'))
        with open(filename / 'metadata.json', 'r') as f:
            self.metadata = json.load(f)

    def __repr__(self):
        """
        """
        extract_shape = lambda x: x.shape if x is not None else 'None'
        extract_dtype = lambda x: x.dtype if x is not None else 'None'

        output = "FrameSequence:\n"
        for k, v in asdict(self).items():
            if k != 'metadata':
                output += f"  {k}: {{'shape': {extract_shape(v)}, 'dtype': {extract_dtype(v)}}}\n"
        output += f"  metadata: {json.dumps(self.metadata, indent=2, separators=(',', ': '))}"
        return output
    

if __name__ == '__main__':
    pass