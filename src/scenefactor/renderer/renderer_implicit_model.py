from dataclasses import dataclass, field

from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig


@dataclass
class ScenefactorModelConfig(SplatfactoModelConfig):
    """
    """
    _target: type = field(default_factory=lambda: ScenefactorModel)


class ScenefactorModel(SplatfactoModel):
    """
    """
    pass