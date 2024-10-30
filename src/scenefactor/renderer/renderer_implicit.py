import json
import yaml
from copy import deepcopy
from glob import glob
from pathlib import Path

import torch
from natsort import natsorted
from omegaconf import OmegaConf
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.scripts import train
from nerfstudio.utils.eval_utils import eval_load_checkpoint
from nerfstudio.pipelines.base_pipeline import VanillaPipeline

from scenefactor.data.common import NumpyTensor
from scenefactor.data.sequence import FrameSequence, save_sequence
from scenefactor.renderer.renderer_implicit_config import ScenefactorMethod
from scenefactor.renderer.renderer_implicit_data_utils import transform_and_scale


def load_pipeline(checkpoint: Path | str, device='cuda') -> VanillaPipeline:
    """
    """
    with open(Path(checkpoint) / 'config.yml') as f:
        config = yaml.unsafe_load(f)
    config.load_dir = checkpoint / 'nerfstudio_models'
    
    # Load transforms
    dataparser = config.pipeline.datamanager.dataparser
    with open(Path(checkpoint) / 'dataparser_transforms.json', 'r') as f:
        transforms = json.load(f)
    dataparser.dataparser_scale = transforms['dataparser_scale']
    dataparser.dataparser_transform = torch.tensor(transforms['dataparser_transform'])

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
        self.output = Path(self.config.output_dir) / self.config.experiment_name / self.config.method_name

        save_sequence(sequence_path, sequence)
        self.sequence = sequence
        self.pipeline = None

    def train(self):
        """
        """
        train.main(self.config)

    def mount(self) -> VanillaPipeline:
        """
        """
        self.pipeline = load_pipeline(self.checkpoint())
        self.pipeline.eval()
    
    def render(self, poses: NumpyTensor['batch', 4, 4]) -> dict:
        """
        """
        poses = torch.from_numpy(poses).float()
        poses = transform_and_scale(
            poses,
            self.pipeline.datamanager.dataparser.config.dataparser_scale,
            self.pipeline.datamanager.dataparser.config.dataparser_transform
        )
        cameras = Cameras(poses, **self.sequence.metadata['camera_params'])
        outputs = self.pipeline.model.get_outputs_for_camera(cameras)
        #for k, v in outputs.items():
        #    print(k, v.shape)
        return outputs

    def checkpoint(self) -> str:
        """
        Returns the latest checkpoint (ordered by time) or raises FileNotFoundError if no checkpoint is found.
        """
        def read_latest(pattern: str):
            return Path(natsorted(glob(pattern))[-1])
        
        if not self.output.exists():
            raise FileNotFoundError(f'No checkpoint found in {self.output}. Make sure to call train() first.')
        return read_latest(str(self.output) + '/*')


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

    #renderer.train()
    renderer.mount()
    image = renderer.render(sequence.poses[0:1])['rgb']
    image = image.cpu().numpy()
    image = (image * 255).astype(int)
    from scenefactor.utils.visualize import visualize_image
    visualize_image(image).save('tests/renderer_implicit.png')
    image_gt = sequence.images[0]
    visualize_image(image_gt).save('tests/renderer_implicit_gt.png')