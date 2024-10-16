from dataclasses import dataclass, field

from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig, VanillaPipeline


@dataclass
class ScenefactorPipelineConfig(VanillaPipelineConfig):
    """
    """
    _target: type = field(default_factory=lambda: ScenefactorPipeline)


class ScenefactorPipeline(VanillaPipeline):
    """
    """
    pass