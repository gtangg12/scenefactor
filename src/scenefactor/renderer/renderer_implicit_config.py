import yaml

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.models.splatfacto import SplatfactoModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig

from scenefactor.renderer.renderer_implicit_data import ScenefactorDataParserConfig, ScenefactorDataManagerConfig

"""
pip install gsplat==1.4.0+pt24cu121 --index-url https://docs.gsplat.studio/whl
"""

ScenefactorMethod = MethodSpecification(
    TrainerConfig(
        method_name='scenefactor',
        steps_per_eval_image=500,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=10000,
        max_num_iterations=10000,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=ScenefactorDataManagerConfig(
                dataparser=ScenefactorDataParserConfig()
            ),
            model=SplatfactoModelConfig(
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
    ),
    description='Implicit renderer for scenefactor.'
)