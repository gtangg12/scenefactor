import os
import shutil
from pathlib import Path
from collections import defaultdict
from termcolor import colored

import torch
from omegaconf import OmegaConf
from trimesh.base import Trimesh

from scenefactor.data.sequence import FrameSequence
from scenefactor.models import ModelInstantMesh
from scenefactor.occlusion_resolver import OcclusionResolver
from scenefactor.sequence_extractor import SequenceExtractor
from scenefactor.sequence_inpainter import SequenceInpainter
from scenefactor.factorization_common import *


def reset_cache(path):
    """
    """
    if Path(path).exists():
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def make_visualization_dir(cache: Path | str, iteration: int = None) -> Path:
    """
    """
    path = Path(cache) / 'visualizations'
    if iteration is not None:
        path = path / f'iteration_{iteration}'
    reset_cache(path)
    return path


class SequenceFactorization:
    """
    """
    def __init__(self, config: OmegaConf):
        """
        """
        self.config = OmegaConf.create(config)
        self.occlusion_cache = Path(self.config.cache) / 'occlusion'
        self.extractor_cache = Path(self.config.cache) / 'extractor'
        self.inpainter_cache = Path(self.config.cache) / 'inpainter'
        self.generator_cache = Path(self.config.cache) / 'generator'

    def __call__(self, sequence: FrameSequence) -> dict[int, Trimesh]:
        """
        """
        reset_cache(self.occlusion_cache)
        reset_cache(self.extractor_cache)
        reset_cache(self.inpainter_cache)
        reset_cache(self.generator_cache)
        occlusion = OcclusionResolver(self.config.occlusion)
        extractor = SequenceExtractor(self.config.extractor)
        inpainter = SequenceInpainter(self.config.inpainter)

        sequence = sequence.clone()

        crops, sequences = defaultdict(dict), {0: sequence.clone()}

        for i in range(self.config.max_iterations):
            occlusion_visualizations = make_visualization_dir(self.occlusion_cache, i) if self.config.visualize_occlusion else None
            extractor_visualizations = make_visualization_dir(self.extractor_cache, i) if self.config.visualize_extractor else None
            inpainter_visualizations = make_visualization_dir(self.inpainter_cache, i) if self.config.visualize_inpainter else None

            frames = occlusion.process_sequence(sequence[::self.config.extractor_step]        , visualizations=occlusion_visualizations)
            images = extractor.process_sequence(sequence[::self.config.extractor_step], frames, visualizations=extractor_visualizations)
            print(colored(f'Iteration {i} extracted {len(images)} mesh image crops, now inpainting sequence', 'green'))
            if len(images) == 0:
                break
            crops.update(images)
            sequence = inpainter.process_sequence(sequence, images.keys(), visualizations=inpainter_visualizations)
            sequences[i + 1] = sequence.clone()
        exit()
        del occlusion # Not enough memory to load all modules at once
        del extractor
        del inpainter
        torch.cuda.empty_cache() # Apparently this is necessary (unlike in InstantMesh/run.py)
        
        generator = ModelInstantMesh(self.config.generator)
        generator_visualizations = make_visualization_dir(self.generator_cache) if self.config.visualize_generator else None
        meshes = {label: generator(crop) for label, crop in crops.items()}
        return meshes


if __name__ == '__main__':
    from scenefactor.data.sequence_reader_replica_vmap import ReplicaVMapFrameSequenceReader
    reader = ReplicaVMapFrameSequenceReader(base_dir='/home/gtangg12/data/replica-vmap', name='office_3')
    sequence = reader.read(slice=(0, -1, 100))
    factorization_config = OmegaConf.load('/home/gtangg12/scenefactor/configs/factorization_replica_vmap.yaml')

    # from scenefactor.data.sequence_reader_graspnet import GraspNetFrameSequenceReader
    # reader = GraspNetFrameSequenceReader(base_dir='/home/gtangg12/data/graspnet', name='scene_0049')
    # sequence = reader.read(slice=(0, -1, 10))
    # factorization_config = OmegaConf.load('/home/gtangg12/scenefactor/configs/factorization_graspnet.yaml')

    factorization_config['cache'] = 'outputs_factorization'
    factorization = SequenceFactorization(factorization_config)

    meshes = factorization(sequence)


'''
Ideas for metrics:

replica (scannet?): look at semantics and get for each scene all the non background objects
    proportion of non background objects extracted to all non background objects
    proportion of background objects extracted to all background objects
    ratio of non background objects to background objects extracted

clip scores of rendered objects vs their gt objects vs images to measure extraction quality

pose errors in registration process

qualitative results of meshes
qualitative results of inpainting
qualitative results of editing/removal in context of whole scene
'''