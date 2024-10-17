from omegaconf import OmegaConf
from trimesh.base import Trimesh

from scenefactor.models import ModelClip, ModelInstantMesh
from scenefactor.utils.geom import *
from scenefactor.utils.visualize import *
from scenefactor.utils.tensor import untile
from scenefactor.factorization_common import *


def metric_multiview_quality(
    model, 
    images: NumpyTensor['batch', 'h', 'w', 3], 
    target: NumpyTensor['h', 'w', 3] | str
) -> float:
    """
    """
    if isinstance(target, str):
        return np.mean([model(image, target) for image in images])
    
    images_embeddings = model.encode_image(images)
    target_embeddings = model.encode_image(target)
    return np.mean(images_embeddings @ target_embeddings.T)


class SequenceGenerator:
    """
    """
    def __init__(self, config: OmegaConf):
        """
        """
        self.config = config
        self.model_generator = ModelInstantMesh(config.model_generator)
        self.model_clip = ModelClip(config.model_clip)

    def __call__(self, images: dict, instance2semantic: dict) -> dict[int, Trimesh]:
        """
        """
        def label2caption(label: int) -> str:
            """
            """
            semantic_label = instance2semantic[label]
            semantic_label = 'well-formed object' if semantic_label == 'unknown' else semantic_label
            return f'An image of a {semantic_label}.'

        meshes = {}
        for iteration, label2crops in images.items():
            for label, candidates in label2crops.items():
                '''
                best_score = 0
                best_tmesh = None
                best_input = None
                best_index = None
                for crop, _, index in candidates:
                    tmesh, mview = self.model_generator(crop, return_multiview=True)
                    score = metric_multiview_quality(self.model_clip, mview, label2caption(label))
                    visualize_tiles(mview, r=3, c=2).save(f'tmp/mview_iter_{iteration}_label_{label}_index_{index}_score_{score:2f}.png')
                    tmesh.export(f'tmp/mesh_iter_{iteration}_label_{label}_index_{index}_score_{score:2f}.obj')
                    if score > best_score:
                        best_score = score
                        best_tmesh = tmesh
                        best_input = crop
                        best_index = index
                meshes[label] = best_tmesh
                meshes[label].export(f'tmp/mesh_iter_{iteration}_label_{label}_index_{best_index}.obj')
                visualize_image(best_input).save(f'tmp/image_iter_{iteration}_label_{label}_index_{index}.png')
                '''
                candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
                for crop, score, index in candidates:
                    tmesh, mview = self.model_generator(crop, return_multiview=True)
                    if self.config.visualize:
                        visualizations_path = f'{self.config.cache}/visualizations'
                        visualizations_tail = f'label_{label}_iter_{iteration}_index_{index}_score_{score:2f}'
                        visualize_image(crop)           .save(f'{visualizations_path}/image_{visualizations_tail}.png')
                        visualize_tiles(mview, r=3, c=2).save(f'{visualizations_path}/mview_{visualizations_tail}.png')
                        tmesh                         .export(f'{visualizations_path}/tmesh_{visualizations_tail}.obj')
                    break
                meshes[label] = tmesh
        return meshes