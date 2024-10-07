import yaml
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np
from glob import glob

from scenefactor.data.sequence import FrameSequence


class FrameSequenceReader(ABC):
    """
    """
    READER_CONFIG = None
    
    def __init__(
        self, 
        base_dir: Path | str, 
        save_dir: Path | str, name: str
    ):
        """
        """
        self.name = name
        self.base_dir = Path(base_dir)
        self.data_dir = Path(base_dir) / name
        self.save_dir = Path(save_dir) / name
        self.metadata = self.load_metadata()

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
        return yaml.safe_load(open(cls.READER_CONFIG, 'r'))


if __name__ == '__main__':
    pass