from dataclasses import dataclass, field

from nerfstudio.fields.nerfacto_field import NerfactoField
from nerfstudio.models.nerfacto import NerfactoModelConfig, NerfactoModel


class ScenefactorField(NerfactoField):
    """
    """
    pass


@dataclass
class ScenefactorModelConfig(NerfactoModelConfig):
    """
    """
    _target: type = field(default_factory=lambda: ScenefactorModel)


class ScenefactorModel(NerfactoModel):
    """
    """
    pass