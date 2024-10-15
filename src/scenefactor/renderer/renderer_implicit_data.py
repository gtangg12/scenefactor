import pickle
from dataclasses import dataclass, field, asdict
from functools import cached_property
from pathlib import Path

import numpy as np
import torch
from nerfstudio.cameras.cameras import Cameras, camera_utils
from nerfstudio.data.dataparsers.base_dataparser import DataParserConfig, DataParser
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig, FullImageDatamanager
from nerfstudio.data.scene_box import SceneBox

from scenefactor.data.common import TorchTensor
from scenefactor.data.sequence import FrameSequence, load_sequence
from scenefactor.renderer.renderer_implicit_data_utils import *


@dataclass
class ScenefactorDataParserOutputs:
    """
    """
    
    """ """
    sequence: FrameSequence
    """ """
    cameras: Cameras
    """ """
    scene_box: SceneBox
    """ """
    dataparser_scale: float = 1.0
    """ """
    dataparser_transform: TorchTensor[3, 4] = torch.eye(4)[:3]
    """ """
    metadata: dict = field(default_factory=dict)

    def save_dataparser_transform(self, path: Path | str):
        """
        Save dataparser transforms for inference use.
        """
        with open(path, 'wb') as f:
            pickle.dump({
                'dataparser_scale'    : self.dataparser_scale,
                'dataparser_transform': self.dataparser_transform
            }, f)


@dataclass
class ScenefactorDataParserConfig(DataParserConfig):
    """
    """
    _target: type = field(default_factory=lambda: ScenefactorDataParser)

    """ """
    sequence_path: Path | str = Path()


class ScenefactorDataParser(DataParser):
    """
    """
    def _generate_dataparser_outputs(self, split='train') -> ScenefactorDataParserOutputs:
        """
        """
        sequence = load_sequence(self.config.sequence_path)
        sequence.metadata['sequence_path'] = self.config.sequence_path

        poses = torch.from_numpy(sequence.poses).float()
        poses, dataparser_transform = \
            camera_utils.auto_orient_and_center_poses(poses, method='up', center_method='poses')
        poses_norm = float(torch.max(torch.abs(poses[:, :3, 3])))
        dataparser_scale = 1 / poses_norm if poses_norm > 0 else 1
        poses[:, :3, 3] *= dataparser_scale

        return ScenefactorDataParserOutputs(
            sequence,
            make_cameras_from_sequence(poses, sequence),
            make_scene_box(),
            dataparser_scale,
            dataparser_transform
        )


class ScenefactorDataset(torch.utils.data.Dataset):
    """
    """
    def __init__(self, dataparser_outputs: ScenefactorDataParserOutputs, *args, **kwargs):
        """
        """
        super().__init__()
        self.sequence  = dataparser_outputs.sequence
        self.cameras   = dataparser_outputs.cameras
        self.scene_box = dataparser_outputs.scene_box
        self.metadata  = dataparser_outputs.metadata
        self.exclude_batch_keys_from_device = [] # do not move to device to save memory

    def __len__(self):
        """
        """
        return len(self.sequence)
    
    def __getitem__(self, index):
        """
        """
        return self.get_data(index)

    def get_data(self, index, **kwargs):
        """
        """
        return {
            'mask': torch.ones(*self.image_dims(index), 1), 
            'image_idx': index,
            'image': self.load_image(index, self.sequence),
        }
    
    def load_image(self, index: int, sequence: FrameSequence):
        """
        """
        return torch.from_numpy(sequence.images[index]) / 255

    def load_depth(self, index: int, sequence: FrameSequence):
        """
        """
        scale = sequence.metadata['depth_scale'] * \
                sequence.metadata['dataparser_scale']
        return torch.from_numpy(sequence.depths[index]) * scale

    def load_cmask(self, index: int, sequence: FrameSequence, name: str):
        """
        """
        return torch.from_numpy(sequence.smasks[index]) if name == 'smask' else \
               torch.from_numpy(sequence.imasks[index])
    
    def image_dims(self, index):
        """ 
        Returns the dataparser specified image dimensions of the datapoint located at `index`.
        """
        return (
            self.cameras[index].height.item(), 
            self.cameras[index].width .item(),
        )


@dataclass
class ScenefactorDataManagerConfig(FullImageDatamanagerConfig):
    """
    """
    _target: type = field(default_factory=lambda: ScenefactorDataManager)


class ScenefactorDataManager(FullImageDatamanager):
    """
    """
    def __init__(self, *args, **kwargs):
        """
        """
        super().__init__(*args, **kwargs)
        # cached_property complicated with inheritance
        self.cached_train = self._load_images('train', cache_images_device=self.config.cache_images)
        self.cached_eval  = self._load_images('eval',  cache_images_device=self.config.cache_images)

    @cached_property
    def dataset_type(self) -> type:
        return ScenefactorDataset