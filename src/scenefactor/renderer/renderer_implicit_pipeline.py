import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch
from omegaconf import OmegaConf
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig, VanillaPipeline
from nerfstudio.utils.eval_utils import eval_load_checkpoint


@dataclass
class ScenefactorPipelineConfig(VanillaPipelineConfig):
    """
    """
    _target: type = field(default_factory=lambda: ScenefactorPipeline)


class ScenefactorPipeline(VanillaPipeline):
    """
    """
    pass


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