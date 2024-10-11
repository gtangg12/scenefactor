import copy
import json
import os
import pickle
from collections import defaultdict, Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from scenefactor.data.common import NumpyTensor
from scenefactor.utils.camera import ray_bundle


@dataclass
class FrameSequence:
    """
    """
    images: NumpyTensor['n', 'h', 'w', 3]
    depths: NumpyTensor['n', 'h', 'w'] = None
    smasks: NumpyTensor['n', 'h', 'w'] = None
    imasks: NumpyTensor['n', 'h', 'w'] = None
    poses : NumpyTensor['n', 4, 4] = None

    metadata: dict = field(default_factory=dict)

    def __len__(self):
        """
        """
        return len(self.images)

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
    with open(filename / 'metadata.pkl', 'wb') as f:
        pickle.dump(sequence.metadata, f)


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
    with open(filename / 'metadata.pkl', 'rb') as f:
        sequence.metadata = pickle.load(f)
    return sequence


def compute_instance2semantic_label_mapping(
    sequence: FrameSequence,
    instance_background: int=None,
    semantic_background: int=None,
) -> dict[int, int]:
    """
    For each instance in the sequence, computes the most common semantic label it is associated with.
    """
    assert sequence.imasks is not None and \
           sequence.smasks is not None
    
    instance2semantic = defaultdict(Counter)
    for imask, smask in zip(sequence.imasks, sequence.smasks):
        for instance_id in np.unique(imask):
            if instance_id == instance_background:
                continue
            match = smask[imask == instance_id]
            if semantic_background is not None:
                match = match[match != semantic_background]
            instance2semantic[instance_id].update(match)
    return {
        k: v.most_common(1)[0][0] for k, v in instance2semantic.items() if len(v)
    }


def sequence_to_pc(sequence: FrameSequence, label: int = None) -> NumpyTensor['n', 3]:
    """
    """
    assert sequence.poses  is not None
    assert sequence.depths is not None
    if label is not None:
        assert sequence.imasks is not None
    
    origins, directions = ray_bundle(sequence.poses, sequence.metadata['camera_params'], norm_directions=False)
    points = origins[:, None, None, :] + directions * sequence.depths[..., None]
    points = points[sequence.depths[..., None].repeat(3, axis=-1) > 0] # remove background points
    if label is not None:
        points = points[sequence.imasks == label]
    return points.reshape(-1, 3)