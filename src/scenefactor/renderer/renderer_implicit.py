from copy import deepcopy
from glob import glob
from pathlib import Path

import torch
from omegaconf import OmegaConf
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.scripts import train

from scenefactor.data.common import NumpyTensor
from scenefactor.data.sequence import FrameSequence, save_sequence
from scenefactor.renderer.renderer_implicit_pipeline import ScenefactorPipelineConfig, ScenefactorPipeline, load_pipeline
from scenefactor.renderer.renderer_implicit_data_utils import transform_and_scale
from scenefactor.renderer.renderer_implicit_data import ScenefactorDataParserConfig, ScenefactorDataManagerConfig
from scenefactor.renderer.renderer_implicit_model import ScenefactorModelConfig


"""
pip install gsplat==1.4.0+pt24cu121 --index-url https://docs.gsplat.studio/whl
"""

TRAINER_CONFIG_TEMPLATE = TrainerConfig(method_name='renderer_implicit',
    steps_per_eval_image=500,
    steps_per_eval_batch=0,
    steps_per_save=2000,
    steps_per_eval_all_images=10000,
    max_num_iterations=10000,
    mixed_precision=False,
    pipeline=ScenefactorPipelineConfig(
        datamanager=ScenefactorDataManagerConfig(
            dataparser=ScenefactorDataParserConfig()
        ),
        model=ScenefactorModelConfig(
            cull_alpha_thresh=0.005,
            densify_grad_thresh=0.0005,
        ),
    ),
    optimizers={
        'means': {
            'optimizer': AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
            'scheduler': ExponentialDecaySchedulerConfig(
                lr_final=1.6e-6,
                max_steps=30000,
            ),
        },
        'features_dc': {
            'optimizer': AdamOptimizerConfig(lr=0.0025, eps=1e-15),
            'scheduler': None,
        },
        'features_rest': {
            'optimizer': AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
            'scheduler': None,
        },
        'opacities': {
            'optimizer': AdamOptimizerConfig(lr=0.05, eps=1e-15),
            'scheduler': None,
        },
        'scales': {
            'optimizer': AdamOptimizerConfig(lr=0.005, eps=1e-15),
            'scheduler': None,
        },
        'quats': {'optimizer': AdamOptimizerConfig(lr=0.001, eps=1e-15), 'scheduler': None},
        'camera_opt': {
            'optimizer': AdamOptimizerConfig(lr=1e-4, eps=1e-15),
            'scheduler': ExponentialDecaySchedulerConfig(
                lr_final=5e-7, max_steps=30000, warmup_steps=1000, lr_pre_warmup=0
            ),
        },
        'bilateral_grid': {
            'optimizer': AdamOptimizerConfig(lr=5e-3, eps=1e-15),
            'scheduler': ExponentialDecaySchedulerConfig(
                lr_final=1e-4, max_steps=30000, warmup_steps=1000, lr_pre_warmup=0
            ),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis='tensorboard',
)


def populate_template(sequence_name: str, sequence_path: Path | str) -> OmegaConf:
    """
    """
    trainer_config = deepcopy(TRAINER_CONFIG_TEMPLATE)
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

    reader = ReplicaVMapFrameSequenceReader(base_dir='/home/gtangg12/data/replica-vmap', name='room_0')
    sequence = reader.read(slice=(0, -1, 10))

    renderer = RendererImplicit(
        sequence, 
        sequence_path='/home/gtangg12/data/scenefactor/replica-vmap/room_0',
        sequence_name='replica_vmap_room_0'
    )
    renderer.train()