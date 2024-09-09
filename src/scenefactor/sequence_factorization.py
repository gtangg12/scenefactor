import multiprocessing as mp
from collections import defaultdict

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
        self.model = ModelInstantMesh(config.model)
    
    def __call__(self, sequence: FrameSequence, instance2semantic: dict[int, int]) -> dict[int, Trimesh]:
        """
        """
        def compute_bmasks(sequence: FrameSequence) -> dict[int, dict[int, NumpyTensor['h', 'w']]]:
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

        sequence_bmasks = compute_bmasks(sequence) # label: index: bmasks

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
        
        meshes = {}
        for label in np.unique(sequence.imasks):
            meshes[label] = process(label)
        exit()
        return meshes


class SequenceInpainting:
    """
    """
    INPAINT_PROMPT = 'Fill in the missing parts of the image.'

    def __init__(self, config: OmegaConf):
        """
        """
        self.config = config
        self.model_inpainter = ModelStableDiffusion(config.model_inpainter)
        self.model_segmenter = ModelSAM2           (config.model.segmenter)

    def __call__(self, sequence: FrameSequence, labels_to_inpaint: set[int]) -> FrameSequence:
        """

        NOTE:: we do not propagate inpainting e.g. using NeRF since image to 3d involves only one frame
        """ 
        def predict_label(index: int, bmask: NumpyTensor['h', 'w']) -> int:
            """
            """
            frame = sequence[index]
            return np.argmax(np.bincount(frame.imask[bmask]))

        def inpaint(index: int, label: int):
            """
            """
            frame = sequence[index]
            bmask = frame.imask == label
            frame.image = self.model_inpainter(self.INPAINT_PROMPT, frame.image, bmask)
            for bmask_inpainted in self.model_segmenter(frame.image):
                if bmask_iou(bmask, bmask_inpainted) > self.config.bmask_iou_threshold:
                    # inpainting operates on tensor views of sequence
                    frame.imask[bmask & bmask_inpainted] = predict_label(index, bmask_inpainted)

        sequence = sequence.clone()
        for label in labels_to_inpaint:
            for i in range(len(sequence)):
                inpaint(i, label)
        return sequence


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
        sequence.metadata['labels'] = set(np.unique(sequence.imask))
        meshes_total = {}
        while True: # either use num iters or until no more new labels extracted
            meshes = self.extractor(sequence, instance2semantic)
            sequence = self.inpainter(sequence, meshes.keys())
            meshes_total.update(meshes)
        return meshes_total


if __name__ == '__main__':
    from scenefactor.data.sequence_reader import ReplicaVMapFrameSequenceReader

    reader = ReplicaVMapFrameSequenceReader('/home/gtangg12/data/replica-vmap', 'room_0', save_dir='/home/gtangg12/data/scenefactor/replica-vmap')
    sequence = reader.read()

    instance2semantic = compute_instance2semantic_label_mapping(sequence, semantic_background=0)

    extractor = SequenceExtraction(OmegaConf.create({
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
    }))
    extractor(sequence, instance2semantic)
    


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