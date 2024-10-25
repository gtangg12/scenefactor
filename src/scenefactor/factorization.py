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


class SequenceFactorization:
    """
    """
    def __init__(self, config: OmegaConf):
        """
        """
        self.config = OmegaConf.create(config)
        self.config.occlusion.cache = Path(self.config.cache) / 'occlusion'
        self.config.extractor.cache = Path(self.config.cache) / 'extractor'
        self.config.inpainter.cache = Path(self.config.cache) / 'inpainter'
        self.config.generator.cache = Path(self.config.cache) / 'generator'

    def __call__(self, sequence: FrameSequence) -> dict[int, Trimesh]:
        """
        """
        reset_cache(self.config.cache)
        occlusion = OcclusionResolver(self.config.occlusion)
        extractor = SequenceExtractor(self.config.extractor)
        inpainter = SequenceInpainter(self.config.inpainter)

        def set_visualizations_dirs(iteration: int):
            """
            """
            occlusion.visualizations_path = Path(occlusion.visualizations_path) / f'iteration_{iteration}'
            extractor.visualizations_path = Path(extractor.visualizations_path) / f'iteration_{iteration}'
            inpainter.visualizations_path = Path(inpainter.visualizations_path) / f'iteration_{iteration}'
            reset_cache(occlusion.visualizations_path)
            reset_cache(extractor.visualizations_path)
            reset_cache(inpainter.visualizations_path)
        
        def pop_visualizations_dirs():
            """
            """
            occlusion.visualizations_path = occlusion.visualizations_path.parent
            extractor.visualizations_path = extractor.visualizations_path.parent
            inpainter.visualizations_path = inpainter.visualizations_path.parent
        
        sequence = sequence.clone()

        crops, sequences = defaultdict(dict), {0: sequence.clone()}
        for i in range(self.config.max_iterations):
            set_visualizations_dirs(i)
            frames = occlusion.process_sequence(sequence)
            images = extractor.process_sequence(sequence, frames)
            print(colored(f'Iteration {i} extracted {len(images)} mesh image crops, now inpainting sequence', 'green'))
            if len(images) == 0:
                break
            crops.update(images)
            sequence = inpainter.process_sequence(sequence, images.keys())
            sequences[i + 1] = sequence.clone()
            pop_visualizations_dirs()
        exit()
        del occlusion # Not enough memory to load all modules at once
        del extractor
        del inpainter
        torch.cuda.empty_cache() # Apparently this is necessary (unlike in InstantMesh/run.py)
        
        generator = ModelInstantMesh(self.config.generator)
        reset_cache(generator)
        meshes = {label: generator(crop) for label, crop in crops.items()}
        return meshes


if __name__ == '__main__':
    from scenefactor.data.sequence_reader_replica_vmap import ReplicaVMapFrameSequenceReader
    reader = ReplicaVMapFrameSequenceReader(base_dir='/home/gtangg12/data/replica-vmap', name='room_0')
    sequence = reader.read(slice=(0, -1, 100))

    # from scenefactor.data.sequence_reader_graspnet import GraspNetFrameSequenceReader
    # reader = GraspNetFrameSequenceReader(base_dir='/home/gtangg12/data/graspnet', name='scene_0001')
    # sequence = reader.read(slice=(0, 100, 25))
    
    config = OmegaConf.load('configs/factorization.yaml')

    factorization_config = OmegaConf.load('/home/gtangg12/scenefactor/configs/factorization.yaml')
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