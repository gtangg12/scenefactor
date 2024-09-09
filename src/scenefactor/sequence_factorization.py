import multiprocessing as mp
import os
from collections import defaultdict
from glob import glob
from pathlib import Path

import cv2
import numpy as np
from omegaconf import OmegaConf
from trimesh.base import Trimesh
from tqdm import tqdm

from scenefactor.data.common import NumpyTensor
from scenefactor.data.sequence import FrameSequence, compute_instance2semantic_label_mapping
from scenefactor.models import ModelInstantMesh, ModelStableDiffusion, ModelSAM2
from scenefactor.utils.geom import *
from scenefactor.utils.colormaps import *
from scenefactor.sequence_factorization_utils import *


class SequenceExtraction:
    """
    """
    def __init__(self, config: OmegaConf):
        """
        """
        self.config = config
        self.model = ModelInstantMesh(config.model) # does not consume gpu memory since calls script internally
    
    def __call__(self, sequence: FrameSequence, instance2semantic: dict[int, int]) -> dict[int, Trimesh]:
        """
        """        
        def compute_bmasks(sequence: FrameSequence):
            """
            """
            bmasks = defaultdict(dict)
            for index, imask in tqdm(enumerate(sequence.imasks), desc='Sequence extraction computing bmasks'):
                for label in np.unique(imask):
                    if label not in instance2semantic:
                        continue
                    semantic_info = sequence.metadata['semantic_info'][instance2semantic[label]]
                    if semantic_info['class'] == 'stuff':
                        continue
                    bmask = imask == label
                    bmask = remove_artifacts(bmask, mode='islands', min_area=self.config.score_hole_area_threshold)
                    bmask = remove_artifacts(bmask, mode='holes'  , min_area=self.config.score_hole_area_threshold)
                    if not np.any(bmask):
                        continue
                    bmasks[label][index] = bmask
            return bmasks
        
        sequence_bmasks = compute_bmasks(sequence) # label: index: bmask

        def compute_bboxes(sequence: FrameSequence) -> list[dict[int, BBox]]:
            """
            """
            boxes = defaultdict(dict)
            for iter, (label, index2bmask) in tqdm(enumerate(sequence_bmasks.items()), desc='Sequence extraction computing bboxes'):
                for index, bmask in index2bmask.items():
                    boxes[label][index] = compute_bbox(bmask)
            return boxes
        
        sequence_bboxes = compute_bboxes(sequence) # label: index: bbox

        def compute_view_score(label: int, index: int) -> float:
            """
            """
            bmask = sequence_bmasks[label][index]
            imask = sequence.imasks[index]
            image = sequence.images[index]
            bbox  = sequence_bboxes[label][index]
            bbox_expanded = resize_bbox(sequence_bboxes[label][index], mult=self.config.score_expand_mult)

            # Condition 1: valid bbox
            if not bbox_check_bounds(bbox_expanded, *bmask.shape):
                return 0, None
            bbox_occupancy = bmask.sum() / bbox_area(bbox)
            if bbox_occupancy < self.config.score_bbox_occupancy_threshold:
                return 0, None
            if bbox_area(bbox) < self.config.score_bbox_min_area:
                return 0, None
            
            # Condition 2: no holes due to occlusions
            holes = compute_holes(bmask)
            if len(holes):
                return 0, None
            
            # Condition 3: no handle non hole occlusion cases
            for label_other in sequence_bboxes:
                if label == label_other:
                    continue
                if index not in sequence_bboxes[label_other]:
                    continue
                bbox_other = sequence_bboxes[label_other][index]
                intersection = bbox_intersection(bbox, bbox_other)
                if intersection is None:
                    continue
                if bbox_area(intersection) < self.config.score_bbox_overlap_threshold:
                    continue
                labels, counts = np.unique(crop(imask, intersection), return_counts=True)
                if labels[np.argmax(counts)] == label:
                    continue
                return 0, None

            score = bbox_occupancy
            image_view = crop(remove_background(image, bmask, background=255), bbox_expanded)
            bmask_view = crop(bmask, bbox_expanded)
            return score, image_view

        def process(label: int) -> Trimesh | None:
            """
            """
            if label not in sequence_bmasks:
                return None
            
            max_score = 0
            max_image = None
            max_index = None
            for index, _ in sequence_bmasks[label].items():
                score, image = compute_view_score(label, index)
                if score > max_score:
                    max_score = score
                    max_image = image
                    max_index = index
            if max_image is None:
                return None
            print(max_score, max_index)
            colormap_image(max_image).save(f'image_view_{label}_{max_index}.png')
            mesh = self.model(max_image)
            mesh.export(f'mesh_{label}.obj')
            return mesh
        
        labels = np.unique(sequence.imasks)
        import random
        random.shuffle(labels)
        labels = [72]
        meshes = {}
        for label in labels:
            meshes[label] = process(label)
            if meshes[label] is not None:
                return meshes
        return meshes


class SequenceInpainting:
    """
    """
    INPAINTING_PROMPT = 'Background'

    def __init__(self, config: OmegaConf):
        """
        """
        self.config = config

    def __call__(self, sequence: FrameSequence, labels_to_inpaint: set[int]) -> FrameSequence:
        """

        NOTE:: we do not propagate inpainting e.g. using NeRF since image to 3d involves only one frame
        """ 
        # Not enough memory to load all models at once
        def call_inpainter(*args, **kwargs):
            return ModelStableDiffusion(self.config.model_inpainter)(*args, **kwargs)
        
        def call_segmenter(*args, **kwargs):
            return ModelSAM2(self.config.model_segmenter)(*args, **kwargs)
        
        def predict_label(index: int, bmask: NumpyTensor['h', 'w']) -> int:
            """
            """
            return np.argmax(np.bincount(sequence.imasks[index][bmask]))

        def inpaint(sequence_updated: FrameSequence, index: int, label: int):
            """
            """
            bmask = sequence_updated.imasks[index] == label
            if not np.any(bmask):
                return
            bmask = cv2.dilate(bmask.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1).astype(bool)
            image = sequence_updated.images[index]
            colormap_image(image).save(f'image_{index}.png')
            colormap_bmask(bmask).save(f'bmask_{index}.png')
            image_updated = call_inpainter(self.INPAINTING_PROMPT, image, bmask)
            masks = call_segmenter(image_updated) 
            colormap_image(image_updated).save(f'image_updated_{index}.png')
            colormap_bmasks(masks).save(f'masks_{index}.png')
            for bmask_sam in masks:
                if bmask_iou(bmask, bmask_sam) > self.config.bmask_iou_threshold:
                    # inpainting operates on tensor views of sequence
                    sequence_updated.imasks[index][bmask & bmask_sam] = predict_label(index, bmask_sam)
            sequence_updated.images[index] = image_updated

        sequence_updated = sequence.clone()
        for label in labels_to_inpaint:
            for i in range(len(sequence)):
                inpaint(sequence_updated, i, label)
            exit()
        return sequence_updated


class SequenceFactorization:
    """
    """
    def __init__(self, config: OmegaConf):
        """
        """
        self.config = config
        self.extractor = SequenceExtraction(config.extractor)
        self.inpainter = SequenceInpainting(config.inpaintor)

    def __call__(self, sequence: FrameSequence) -> dict[int, Trimesh]:
        """
        """
        instance2semantic = compute_instance2semantic_label_mapping(sequence, semantic_background=0)

        sequence = sequence.clone()
        meshes_total = {}
        for i in range(self.config.max_iterations):
            meshes = self.extractor(sequence, instance2semantic)
            if len(meshes) == 0:
                break
            sequence = self.inpainter(sequence, meshes.keys())
            meshes_total.update(meshes)
        return meshes_total


if __name__ == '__main__':
    from scenefactor.data.sequence_reader import ReplicaVMapFrameSequenceReader

    reader = ReplicaVMapFrameSequenceReader(
        base_dir='/home/gtangg12/data/replica-vmap', 
        save_dir='/home/gtangg12/data/scenefactor/replica-vmap', 
        name='room_0',
    )
    sequence = reader.read()

    instance2semantic = compute_instance2semantic_label_mapping(sequence, semantic_background=0)

    extractor_config = OmegaConf.create({
        'model': {
            'script_path': 'third_party/InstantMesh/run.py',
            'model_config_path': 'configs/instant-mesh-large.yaml',
            'image_path': 'examples/input.png',
            'tmesh_path': 'outputs/instant-mesh-large/meshes/input.obj'
        },
        'score_expand_mult': 1.25,
        'score_hole_area_threshold': 128,
        'score_bbox_min_area': 1024,
        'score_bbox_occupancy_threshold': 0.6,
        'score_bbox_overlap_threshold': 64,
        'cache_dir': reader.save_dir / 'sequence_factorization_extraction' 
    })
    inpainter_config = OmegaConf.create({
        'model_inpainter': {
            'checkpoint': 'benjamin-paine/stable-diffusion-v1-5-inpainting'
        },
        'model_segmenter': {
            'checkpoint': 'checkpoints/sam2_hiera_large.pt', 
            'model_config': 'sam2_hiera_l.yaml',
            'mode': 'auto'
        },
        'bmask_iou_threshold': 0.5,
        'cache_dir': reader.save_dir / 'sequence_factorization_inpainting'
    })
    sequence_factorization = SequenceFactorization(OmegaConf.create({
        'extractor': extractor_config,
        'inpaintor': inpainter_config,
        'max_iterations': 1,
    }))
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