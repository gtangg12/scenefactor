import json
import os
import yaml
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np
from glob import glob
from natsort import natsorted

from scenefactor.data.common import NumpyTensor, CONFIGS_DIR
from scenefactor.data.sequence import FrameSequence, save_sequence, load_sequence


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
        self.data_dir = Path(base_dir) / name
        self.save_dir = Path(save_dir) / name
        self.metadata = self.load_metadata()

    def sequence_names(self):
        """
        Returns a list of sequence names.
        """
        return list(sorted(glob(str(self.data_dir))))

    @abstractmethod
    def read(self, slice=(0, -1, 1), override=False) -> FrameSequence:
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
    READER_CONFIG = CONFIGS_DIR / 'sequence_reader_replica_vmap.yaml'

    def __init__(
        self, 
        base_dir: Path | str, 
        save_dir: Path | str, name: str, track='00'
    ):
        """
        """
        super().__init__(base_dir, save_dir, name)

        assert track in ['00', '01']
        self.track = track
        self.data_dir = self.data_dir / 'imap' / track
        self.save_dir = self.save_dir / track

    def read(self, slice=(0, -1, 20), override=False) -> FrameSequence:
        """
        """
        if self.save_dir.exists() and not override:
            return load_sequence(self.save_dir)

        def read_filenames(pattern: str) -> list[str]:
            """
            """
            filenames = natsorted(glob(f'{self.data_dir}/{pattern}'))
            filenames = filenames[slice[0]:slice[1]:slice[2]]
            return filenames
        
        def read_poses() -> NumpyTensor['n', 4, 4]:
            """
            """
            poses = np.loadtxt(self.data_dir / 'traj_w_c.txt', delimiter=' ')
            poses = poses.reshape(-1, 4, 4)
            poses = poses[slice[0]:slice[1]:slice[2]]
            poses = poses @ np.array(self.metadata['pose_axis_transform'])
            return poses
        
        def read_semantic_info() -> dict[int, dict]:
            """
            """
            filename = f'{self.base_dir}/{self.name}/habitat/info_preseg_semantic.json'
            semantic_info = json.load(open(filename, 'r'))['classes']
            semantic_classes_things = self.metadata.pop('semantic_classes_things')
            return {
                data['id']: \
                    {'name': data['name'], 'class': 'thing' if data['name'] in semantic_classes_things else 'stuff'}
                for data in semantic_info
            }
        
        image_filenames = read_filenames('rgb/*.png')
        depth_filenames = read_filenames('depth/*.png')
        smask_filenames = read_filenames('semantic_class/semantic_class_*.png')
        imask_filenames = read_filenames('semantic_instance/semantic_instance_*.png')
        
        poses = read_poses()

        self.metadata['semantic_info'] = read_semantic_info()
        
        sequence = FrameSequence(
            poses=poses,
            images=np.array([self.load_image(f) for f in image_filenames]),
            depths=np.array([self.load_depth(f, scale=self.metadata['depth_scale']) for f in depth_filenames]),
            smasks=np.array([self.load_smask(f) for f in smask_filenames]),
            imasks=np.array([self.load_imask(f) for f in imask_filenames]),
            metadata=self.metadata
        )
        save_sequence(self.save_dir, sequence)
        return sequence


class ScanNetFrameSequenceReader(FrameSequenceReader):
    """
    """
    READER_CONFIG = CONFIGS_DIR / 'sequence_reader_scannet.yaml'

    def read(self, slice=(0, -1, 20), override=False) -> FrameSequence:
        """
        """
        data_dir = self.base_dir / self.name
        save_dir = self.save_dir / self.name if self.save_dir is not None else None
        if save_dir is not None and save_dir.exists() and not override:
            return load_sequence(save_dir)
        
        def read_filenames(pattern: str) -> list[str]:
            """
            """
            filenames = natsorted(glob(f'{data_dir}/{pattern}'))
            filenames = filenames[slice[0]:slice[1]:slice[2]]
            return filenames

        def read_poses() -> NumpyTensor['n', 4, 4]:
            """
            """
            poses = []
            for filename in read_filenames('pose/*.txt'):
                with open(filename, 'r') as f:
                    pose = np.loadtxt(f).reshape(4, 4)
                poses.append(pose)
            poses = np.stack(poses)
            poses = poses[slice[0]:slice[1]:slice[2]]
            poses = poses @ np.array(self.metadata['pose_axis_transform'])
            return poses
        
        def read_semantic_info() -> dict[int, dict]:
            """
            """
            semantic_classes_stuffs = set(self.metadata.pop('semantic_classes_stuffs'))
            semantic_classes_things = set(self.metadata.pop('semantic_classes_things'))
            return {
                label: {'class': 'thing' if label in semantic_classes_things else 'stuff'}
                for label in semantic_classes_stuffs + \
                             semantic_classes_things
            }
    
        image_filenames = read_filenames('color/*.jpg')
        depth_filenames = read_filenames('depth/*.png')
        smask_filenames = read_filenames('semantics/*.png')
        imask_filenames = read_filenames('instance/*.png')

        poses = read_poses()

        self.metadata['semantic_info'] = read_semantic_info()

        sequence = FrameSequence(
            poses=poses,
            images=np.array([self.load_image(f) for f in image_filenames]),
            depths=np.array([self.load_depth(f) for f in depth_filenames]),
            smasks=np.array([self.load_smask(f) for f in smask_filenames]),
            imasks=np.array([self.load_imask(f) for f in imask_filenames]),
            metadata=self.metadata
        )
        if save_dir is not None:
            save_sequence(save_dir, sequence)
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