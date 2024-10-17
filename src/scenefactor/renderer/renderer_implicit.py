import pickle
from copy import deepcopy
from glob import glob
from pathlib import Path

import torch
from omegaconf import OmegaConf
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.scripts import train
from nerfstudio.utils.eval_utils import eval_load_checkpoint

from scenefactor.data.common import NumpyTensor
from scenefactor.data.sequence import FrameSequence, save_sequence
from scenefactor.renderer.renderer_implicit_config import ScenefactorMethod
from scenefactor.renderer.renderer_implicit_pipeline import ScenefactorPipeline
from scenefactor.renderer.renderer_implicit_data_utils import transform_and_scale


def load_pipeline(checkpoint: Path | str, device='cuda') -> ScenefactorPipeline:
    """
    """
    config = OmegaConf.load(Path(checkpoint) / 'config.yaml')
    
    # Load transforms
    dataparser = config.pipeline.datamanager.dataparser
    transforms = pickle.load(open(dataparser.sequence_path / 'transforms.pkl', 'rb'))
    dataparser.update(**transforms)
    
    # Mount checkpoint on pipeline
    pipeline = config.pipeline.setup(device=device)
    eval_load_checkpoint(config, pipeline)
    return pipeline


def populate_template(sequence_name: str, sequence_path: Path | str) -> OmegaConf:
    """
    """
    trainer_config = deepcopy(ScenefactorMethod.config)
    trainer_config.experiment_name = sequence_name
    trainer_config.pipeline.datamanager.dataparser.sequence_path = sequence_path
    return trainer_config


class RendererImplicit:
    """
    """
    def __init__(
        self,
        sequence: FrameSequence,
        sequence_name: str, 
        sequence_path: Path | str
    ):
        """
        """
        self.config = populate_template(sequence_name, sequence_path)
        self.output = Path(self.config.output_dir) / self.config.method_name / self.config.experiment_name

        save_sequence(sequence_path, sequence)
        self.sequence = sequence
        self.pipeline = None

    def train(self) -> ScenefactorPipeline:
        """
        """
        train.main(self.config)
        self.pipeline = load_pipeline(self.checkpoint())
        return self.pipeline

    def render(self, poses: NumpyTensor['batch', 4, 4]) -> dict:
        """
        """
        poses = torch.from_numpy(poses)
        poses = transform_and_scale(
            poses,
            self.pipeline.datamanager.dataparser.dataparser_scale,
            self.pipeline.datamanager.dataparser.dataparser_transform
        )
        cameras = Cameras(poses, **self.sequence.metadata['camera_params'])
        outputs = self.pipeline.model.get_outputs_for_camera(cameras)
        return outputs

    def checkpoint(self) -> str:
        """
        Returns the latest checkpoint (ordered by time) or raises FileNotFoundError if no checkpoint is found.
        """
        path = self.output / 'nerfstudio_models'
        if path.exists():
            return sorted(glob(path))[-1]
        raise FileNotFoundError(f'No checkpoint found in {path}')


if __name__ == '__main__':
    from scenefactor.data.sequence_reader_replica_vmap import ReplicaVMapFrameSequenceReader
    from scenefactor.data.sequence_reader_graspnet import GraspNetFrameSequenceReader

    # reader = ReplicaVMapFrameSequenceReader(base_dir='/home/gtangg12/data/replica-vmap', name='room_0')
    # sequence = reader.read(slice=(0, -1, 5))
    # sequence = sequence.rescale(0.5)
    # renderer = RendererImplicit(
    #     sequence, 
    #     sequence_path='/home/gtangg12/data/scenefactor/replica-vmap/room_0',
    #     sequence_name='replica_vmap_room_0'
    # )

    reader = GraspNetFrameSequenceReader(base_dir='/home/gtangg12/data/graspnet', name='scene_0000')
    sequence = reader.read(slice=(0, -1, 1))
    sequence = sequence.rescale(0.5)
    renderer = RendererImplicit(
        sequence, 
        sequence_path='/home/gtangg12/data/scenefactor/graspnet/scene_0000',
        sequence_name='graspnet_scene_0000'
    )

    renderer.train()