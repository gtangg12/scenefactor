from scenefactor.models.model_instant_mesh import ModelInstantMesh
from scenefactor.models.model_stable_diffusion import ModelStableDiffusion
from scenefactor.models.model_lama import ModelLama
from scenefactor.models.model_grounding_dino import ModelGroundingDino
from scenefactor.models.model_ram import ModelRam
from scenefactor.models.model_sam import ModelSam
from scenefactor.models.model_loftr import ModelLoFTR

ImageTo3DModel = ModelInstantMesh
ImageInpaintingModel = ModelStableDiffusion | ModelLama