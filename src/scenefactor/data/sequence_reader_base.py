import yaml
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from glob import glob

from scenefactor.data.common import NumpyTensor, CONFIGS_DIR
from scenefactor.data.sequence import FrameSequence


class FrameSequenceReader(ABC):
    """
    """
    READER_CONFIG = None
    
    def __init__(self, base_dir: Path | str, name: str):
        """
        """
        self.name = name
        self.base_dir = Path(base_dir)
        self.data_dir = Path(base_dir) / name
        self.metadata = self.load_metadata()
        self.metadata['sequence_name'] = name
        self.metadata['sequence_base'] = base_dir

    def sequence_names(self):
        """
        Returns a list of sequence names.
        """
        return list(sorted(glob(str(self.data_dir))))

    @abstractmethod
    def read(self, slice=(0, -1, 1)) -> FrameSequence:
        """
        """
        pass
    
    @classmethod
    def load_image(cls, filename: Path | str, resize: tuple[int, int]=None):
        """
        Default function to load and resize RGB image.
        """
        image = cv2.imread(str(filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, resize, interpolation=cv2.INTER_LINEAR) if resize else image
        return image

    @classmethod
    def load_depth(cls, filename: Path | str, resize: tuple[int, int]=None, scale: float=1.0):
        """
        Default function to load and resize/scale depth map.
        """
        depth = cv2.imread(str(filename), cv2.IMREAD_UNCHANGED).astype(np.int32)
        depth = cv2.resize(depth, resize, interpolation=cv2.INTER_NEAREST) if resize else depth
        depth = depth.astype(np.float32) * scale
        return depth
    
    @classmethod
    def load_smask(cls, filename: Path | str, resize: tuple[int, int]=None):
        """
        Default function to load and resize semantic label mask.
        """
        smask = cv2.imread(filename, cv2.IMREAD_UNCHANGED).astype(np.int32)
        smask = cv2.resize(smask, resize, interpolation=cv2.INTER_NEAREST) if resize else smask
        return smask

    @classmethod
    def load_imask(cls, filename: Path | str, resize: tuple[int, int]=None):
        """
        Default function to load and resize instance label mask.
        """
        imask = cv2.imread(filename, cv2.IMREAD_UNCHANGED).astype(np.int32)
        imask = cv2.resize(imask, resize, interpolation=cv2.INTER_NEAREST) if resize else imask
        return imask
    
    @classmethod
    def load_metadata(cls):
        """
        Default function to load metadata from yaml config.
        """
        assert cls.READER_CONFIG is not None, f'DATACONFIG is not defined for class {cls.__name__}.'
        metadata = yaml.safe_load(open(CONFIGS_DIR / cls.READER_CONFIG, 'r'))
        return metadata if metadata is not None else {}


def instance_to_most_common_semantic(
    imasks: NumpyTensor['n', 'h', 'w'],
    smasks: NumpyTensor['n', 'h', 'w'],
    instance_background: int=None,
    semantic_background: int=None
) -> dict[int, int]:
    """
    For each instance in the sequence, computes the most common semantic label it is associated with.
    """    
    instance2semantic = defaultdict(Counter)
    for imask, smask in zip(imasks, smasks):
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


def extract_instances_by_semantics(
    imasks: NumpyTensor['n', 'h', 'w'],
    labels: list[int], 
    instance2semantic: dict[int, int], 
    instance_background=0
) -> NumpyTensor['n', 'h', 'w']:
    """
    """
    labels = set(labels)
    for imask in imasks:
        for instance_id, semantic_id in instance2semantic.items():
            if semantic_id not in labels:
                imask[imask == instance_id] = instance_background
    return imasks


if __name__ == '__main__':
    pass