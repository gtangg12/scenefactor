import copy
import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from src.scenefactor.data.common import NumpyTensor


@dataclass
class FrameSequence:
    """
    """
    images: NumpyTensor['n', 'h', 'w', 3] = None
    depths: NumpyTensor['n', 'h', 'w'] = None
    smasks: NumpyTensor['n', 'h', 'w'] = None
    imasks: NumpyTensor['n', 'h', 'w'] = None
    poses : NumpyTensor['n', 4, 4] = None

    metadata: dict = field(default_factory=dict)

    def __len__(self):
        """
        """
        return len(self.image)

    def __getitem__(self, index: int):
        """
        Returns view of FrameSequence object at index.
        """
        return FrameSequence(**{k: tensor[index] for k, tensor in asdict(self).items()})

    def __slice__(self, index1: int, index2: int, inc: int):
        """
        Returns view slice of FrameSequence object from index1 to index2.
        """
        return FrameSequence(**{k: tensor[index1:index2:inc] for k, tensor in asdict(self).items()})
    
    def __getitem__(self, index: int | slice):
        """
        """
        if isinstance(index, int):
            return FrameSequence(**{
                k: v[index]
                if isinstance(v, np.ndarray) else v for k, v in asdict(self).items()
            })
        if isinstance(index, slice):
            return FrameSequence(**{
                k: v[index.start:index.stop:index.step] 
                if isinstance(v, np.ndarray) else v for k, v in asdict(self).items()
            })
        raise ValueError('Index must be an integer or slice')
    
    def clone(self):
        """
        """
        return copy.deepcopy(self)

    def __repr__(self):
        """
        """
        extract_shape = lambda x: x.shape if x is not None else 'None'
        extract_dtype = lambda x: x.dtype if x is not None else 'None'

        stats = defaultdict(int)
        for k, v in asdict(self).items():
            if k == 'metadata':
                continue
            stats['tname_len'] = max(len(k), stats['tname_len'])
            stats['shape_len'] = max(len(str(extract_shape(v))), stats['shape_len'])
            stats['dtype_len'] = max(len(str(extract_dtype(v))), stats['dtype_len'])

        def pad(s, l):
            return s + ' ' * (l - len(s))
        
        output = "FrameSequence:\n"
        for k, v in asdict(self).items():
            if k == 'metadata':
                continue
            tname = pad(k, stats['tname_len'])
            shape = pad(str(extract_shape(v)), stats['shape_len'])
            dtype = pad(str(extract_dtype(v)), stats['dtype_len'])
            output += f"    {tname}: {{'shape': {shape}, 'dtype': {dtype}}}\n"
        output += f"    metadata: {json.dumps(self.metadata, indent=4, separators=(',', ': '))}"
        return output
    

def save_sequence(filename: Path | str, sequence: FrameSequence) -> None:
    """
    Saves data from a FrameSequence object to disk.
    """
    filename = Path(filename)

    os.makedirs(filename, exist_ok=True)
    for k, v in asdict(sequence).items():
        if isinstance(v, np.ndarray):
            np.save(filename / f'{k}.npy', v)
    with open(filename / 'metadata.json', 'w') as f:
        json.dump(sequence.metadata, f)


def load_sequence(filename: Path | str) -> FrameSequence:
    """
    Loads data from disk into a FrameSequence object, overwriting existing data.
    """
    filename = Path(filename)

    sequence = FrameSequence() # empty sequence
    for k, _ in asdict(sequence).items():
        if k == 'metadata':
            continue
        setattr(sequence, k, np.load(filename / f'{k}.npy'))
    with open(filename / 'metadata.json', 'r') as f:
        sequence.metadata = json.load(f)
    return sequence