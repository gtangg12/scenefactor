import os
from collections import defaultdict
from termcolor import colored

import torch
from omegaconf import OmegaConf
from trimesh.base import Trimesh

from scenefactor.data.sequence import FrameSequence, compute_instance2semantic_label_mapping
from scenefactor.factorization_extractor import SequenceExtractor
from scenefactor.factorization_inpainter import SequenceInpainter
from scenefactor.factorization_generator import SequenceGenerator
from scenefactor.factorization_utils import *


class SequenceFactorization:
    """
    """
    def __init__(self, config: OmegaConf):
        """
        """
        self.config = config

    def __call__(self, sequence: FrameSequence) -> dict[int, Trimesh]:
        """
        """
        sequence = sequence.clone()

        instance2semantic = compute_instance2semantic_label_mapping(
            sequence, 
            instance_background=INSTANCE_BACKGROUND, 
            semantic_background=SEMANTIC_BACKGROUND
        )
        
        extractor = SequenceExtractor(self.config.extractor)
        inpainter = SequenceInpainter(self.config.inpainter)

        images_total, sequences_total = defaultdict(dict), {0: sequence.clone()}
        for i in range(self.config.max_iterations):
            images = extractor(sequence, instance2semantic, iteration=i)
            print(colored(f'Iteration {i} extracted {len(images)} mesh image crops, now inpainting sequence', 'green'))
            if len(images) == 0:
                break
            sequence = inpainter(sequence, images.keys(), iteration=i)
            images_total[i], sequences_total[i] = images, sequence.clone()
        exit()
        del extractor, inpainter # Not enough memory to load all modules at once
        torch.cuda.empty_cache() # Apparently this is necessary (unlike in InstantMesh/run.py)
        
        generator = SequenceGenerator(self.config.generator)
        meshes_total = generator(images_total, instance2semantic)
        return meshes_total


if __name__ == '__main__':
    from scenefactor.data.sequence_reader import ReplicaVMapFrameSequenceReader

    reader = ReplicaVMapFrameSequenceReader(
        base_dir='/home/gtangg12/data/replica-vmap', 
        save_dir='/home/gtangg12/data/scenefactor/replica-vmap', 
        name='room_0',
    )
    sequence = reader.read(slice=(0, -1, 200), override=True)

    instance2semantic = compute_instance2semantic_label_mapping(sequence, semantic_background=0)

    sequence_factorization_config = OmegaConf.load('/home/gtangg12/scenefactor/configs/factorization_replica_vmap.yaml')
    sequence_factorization_config['cache'] = reader.save_dir / 'factorization'
    sequence_factorization = SequenceFactorization(sequence_factorization_config)
    if os.path.exists('tmp'):
        import shutil
        shutil.rmtree('tmp')
    os.makedirs('tmp')
    meshes = sequence_factorization(sequence)


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