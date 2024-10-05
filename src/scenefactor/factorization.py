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

    extractor_config = {
        'score_expand_mult': 1.25,
        'score_hole_area_threshold': 128,
        'score_bbox_min_area': 512,
        'score_bbox_occupancy_threshold': 0.1,
        'score_bbox_overlap_threshold': 64,
        'score_threshold': 0.4,
        'score_topk': 5,
        'model_clip': {'name': 'ViT-B/32', 'temperature': 0.1},
    }
    inpainter_config = {
        'model_inpainter': {
            'checkpoint': '/home/gtangg12/scenefactor/checkpoints/big-lama'
        },
        'model_segmenter': {
            'ram': {
                'checkpoint': '/home/gtangg12/scenefactor/checkpoints/ram_plus_swin_large_14m.pth'
            },
            'grounding_dino': {
                'checkpoint'       : '/home/gtangg12/scenefactor/checkpoints/groundingdino_swinb_cogcoor.pth',
                'checkpoint_config': '/home/gtangg12/scenefactor/third_party/GroundingDino/groundingdino/config/GroundingDINO_SwinB_cfg.py',
                'bbox_threshold': 0.25,
                'text_threshold': 0.25,
            },
            'sam_pred': {
                'checkpoint': '/home/gtangg12/scenefactor/checkpoints/sam_vit_h_4b8939.pth',
                'mode': 'pred',
                'engine_config': {}
            },
            'sam_auto': {
                'checkpoint': '/home/gtangg12/scenefactor/checkpoints/sam_vit_h_4b8939.pth',
                'mode': 'auto',
                'engine_config': {}
            },
            'include_sam_auto': True,
            'min_area': 256,
        },
    }
    generator_config = {
        'model_generator': {
            'model_config': 'configs/instant-mesh-large.yaml',
            'input_path': 'examples/input.png',
            'tmesh_path': 'outputs/instant-mesh-large/meshes/input.obj'
        },
        'model_clip': {'name': 'ViT-B/32', 'temperature': 0.1},
    }
    sequence_factorization = SequenceFactorization(OmegaConf.create({
        'extractor': extractor_config,
        'inpainter': inpainter_config,
        'generator': generator_config,
        'max_iterations': 10,
    }))

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