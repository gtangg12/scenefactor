import yaml
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from glob import glob
from natsort import natsorted
from omegaconf import OmegaConf

from scenefactor.data.common import NumpyTensor, TorchTensor
from scenefactor.data.sequence import FrameSequence, save_sequence, load_sequence


CONFIGS_DIR = Path(__file__).resolve().parent / '../../../configs'


class FrameSequenceReader(ABC):
    """
    """
    READER_CONFIG = None
    
    def __init__(self, data_dir: Path | str):
        """
        """
        self.data_dir = Path(data_dir)
        self.metadata = self.load_metadata()

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


class ReplicaVMapFrameSequenceReader(FrameSequenceReader):
    """
    """
    READER_CONFIG = CONFIGS_DIR / 'reader_replica_vmap.yaml'

    def read(self, track='00', slice=(0, -1, 20)) -> FrameSequence:
        """
        """
        assert track in ['00', '01']

        def read_filenames(pattern: str) -> list[str]:
            """
            """
            return natsorted(glob(f'{self.data_dir}/imap/{track}/{pattern}'))[slice[0]:slice[1]:slice[2]]
        
        image_filenames = read_filenames('rgb/*.png')
        depth_filenames = read_filenames('depth/*.png')
        smask_filenames = read_filenames('semantic_class/semantic_class_*.png')
        imask_filenames = read_filenames('semantic_instance/semantic_instance_*.png')

        poses = np.loadtxt(self.data_dir / f'imap/{track}/traj_w_c.txt', delimiter=' ').reshape(-1, 4, 4)
        poses = poses[slice[0]:slice[1]:slice[2]]
        poses = poses @ np.array(self.metadata['pose_axis_transform'])
        
        sequence = FrameSequence(
            poses=poses,
            images=np.array([self.load_image(f) for f in image_filenames]),
            depths=np.array([self.load_depth(f, scale=self.metadata['depth_scale']) for f in depth_filenames]),
            smasks=np.array([self.load_smask(f) for f in smask_filenames]),
            imasks=np.array([self.load_imask(f) for f in imask_filenames]),
            metadata=self.metadata
        )
        return sequence


if __name__ == '__main__':
    reader = ReplicaVMapFrameSequenceReader('/home/gtangg12/data/replica-vmap/office_0')
    sequence = reader.read()
    print(sequence)

    save_sequence('tests/sequence', sequence)
    sequence = load_sequence('tests/sequence')
    print(sequence)

    sequence_item = sequence[0]
    print(sequence_item)

    sequence_slice = sequence[0:10:2]
    print(sequence_slice)
    print(sequence)

    sequence2 = sequence.clone()
    sequence = None
    print(sequence)
    print(sequence2)