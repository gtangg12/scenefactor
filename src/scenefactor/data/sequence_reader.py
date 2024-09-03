from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from glob import glob
from natsort import natsorted
from omegaconf import OmegaConf

from scenefactor.data.common import NumpyTensor, TorchTensor
from scenefactor.data.sequence import FrameSequence


class FrameSequenceReader(ABC):
    """
    """
    READER_CONFIG = None

    @abstractmethod
    def read(self, filename: Path | str, slice=(0, -1, 1)) -> FrameSequence:
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
        return dict(OmegaConf.load(cls.READER_CONFIG))


class ReplicaVMapFrameSequenceReader(FrameSequenceReader):
    """
    """
    READER_CONFIG = '../../../configs/reader_replica_vmap.yaml'

    def read(self, filename: Path | str, track='00', slice=(0, -1, 1)) -> FrameSequence:
        """
        """
        assert track in ['00', '01']

        metadata = self.load_metadata()

        image_filenames = natsorted(glob(f'{self.datadir}/imap/{track}/rgb/*.png'))
        depth_filenames = natsorted(glob(f'{self.datadir}/imap/{track}/depth/*.png'))
        smask_filenames = natsorted(glob(f'{self.datadir}/imap/{track}/semantic_class/semantic_class_*.png'))
        imask_filenames = natsorted(glob(f'{self.datadir}/imap/{track}/semantic_instance/semantic_instance_*.png'))

        poses = np.loadtxt(self.datadir / f'imap/{track}/traj_w_c.txt', delimiter=' ').reshape(-1, 4, 4)
        poses = poses @ np.array(metadata['pose_axis_transform'])
        
        sequence = FrameSequence(
            poses=poses,
            images=np.array([self.load_image(f) for f in image_filenames]),
            depths=np.array([self.load_depth(f, scale=metadata['depth_scale']) for f in depth_filenames]),
            smasks=np.array([self.load_smask(f) for f in smask_filenames]),
            imasks=np.array([self.load_imask(f) for f in imask_filenames]),
            metadata=metadata
        )
        sequence = sequence[slice[0]:slice[1]:slice[2]]
        return sequence
    

if __name__ == '__main__':
    pass