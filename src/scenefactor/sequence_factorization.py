import multiprocessing as mp
import os
from collections import defaultdict
from glob import glob
from pathlib import Path
from termcolor import colored

import cv2
import numpy as np
from omegaconf import OmegaConf
from trimesh.base import Trimesh
from tqdm import tqdm

from scenefactor.data.common import NumpyTensor
from scenefactor.data.sequence import FrameSequence, compute_instance2semantic_label_mapping
from scenefactor.models import ModelInstantMesh, ModelLama, ModelSamGrounded, Clip
from scenefactor.utils.geom import *
from scenefactor.utils.colormaps import *
from scenefactor.sequence_factorization_utils import *


SEMANTIC_BACKGROUND = 0
INSTANCE_BACKGROUND = 0


class SequenceExtraction:
    """
    """
    def __init__(self, config: OmegaConf):
        """
        """
        self.config = config
        self.model_clip = Clip('ViT-B/32')
    
    def __call__(self, sequence: FrameSequence, instance2semantic: dict[int, int], iteration: int) -> dict[int, NumpyTensor['h', 'w', 3]]:
        """
        """
        sequence_bmasks, sequence_bboxes = \
            compute_sequence_bmasks_bboxes(sequence, instance2semantic, min_area=128, background=0)

        def compute_view_score(label: int, index: int, alpha=0.5) -> float:
            """
            """
            bmask = sequence_bmasks[label][index]
            imask = sequence.imasks[index]
            image = sequence.images[index]
            bbox  = sequence_bboxes[label][index]
            bbox_expanded      = resize_bbox(sequence_bboxes[label][index], mult=self.config.score_expand_mult)
            bbox_expanded_test = resize_bbox(bbox_expanded                , mult=self.config.score_expand_mult) # check bbox should be larger than expanded bbox to avoid (literal) edge cases

            # Condition 1: valid bbox
            if not bbox_check_bounds(bbox_expanded_test, *bmask.shape): 
                return 0, None
            bbox_occupancy = bmask.sum() / bbox_area(bbox)
            if bbox_occupancy < self.config.score_bbox_occupancy_threshold:
                return 0, None
            if bbox_area(bbox) < self.config.score_bbox_min_area:
                return 0, None
            
            # Condition 2: no holes due to occlusions
            holes = compute_holes(bmask)
            for _, area in holes:
                if area < self.config.score_hole_area_threshold:
                    continue
                return 0, None

            # Condition 3: handle non-hole occlusion cases
            for label_other in sequence_bmasks:
                if label == label_other:
                    continue
                if index not in sequence_bmasks[label_other]:
                    continue
                # test bmask overlap
                bmask_other = sequence_bmasks[label_other][index]
                intersection = dialate_bmask(bmask, 5) & dialate_bmask(bmask_other, 5)
                if not np.any(intersection):
                    continue
                # test occupancy in bbox intersection
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

            image_view = crop(remove_background(image, bmask, background=255), bbox_expanded)
            semantic_label = instance2semantic[label]
            semantic_label = sequence.metadata['semantic_info'][semantic_label]['name']
            semantic_score = self.model_clip(image_view, f'An image of a {semantic_label}.')
            combined_score = alpha * bbox_occupancy + (1 - alpha) * semantic_score
            print(f'BBox occupancy: {bbox_occupancy:.2f}, Semantic score: {semantic_score:.2f}, Combined score: {combined_score:.2f}, {semantic_label}')
            return combined_score, image_view

        def process(label: int) -> Trimesh | None:
            """
            """
            sequence_labels = sequence_bmasks.keys()
            if label not in sequence_labels:
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
            
            print(colored(f'Extracting label {label}', 'green'))
            colormap_image(max_image).save(f'tmp/image_view_label_{label}_iter_{iteration}_index_{max_index}.png')
            return max_image

        labels = np.unique(sequence.imasks)
        images = {}
        for label in labels:
            if label == 0:
                continue
            image = process(label)
            if image is not None:
                images[label] = image
        return images


class SequenceInpainting:
    """
    """
    def __init__(self, config: OmegaConf):
        """
        """
        self.config = config
        self.model_inpainter = ModelLama(config.model_inpainter)
        self.model_segmenter = ModelSamGrounded(config.model_segmenter)

    def __call__(self, sequence: FrameSequence, labels_to_inpaint: set[int], iteration: int) -> FrameSequence:
        """
        NOTE:: we do not propagate inpainting e.g. using NeRF since image to 3d involves only one frame
        """
        sequence_updated = sequence.clone()

        def predict_label(
            imask_input: NumpyTensor['h', 'w'],
            bmask_input: NumpyTensor['h', 'w'],
            imask_paint: NumpyTensor['h', 'w'],  
            bmask_label: NumpyTensor['h', 'w'], radius=10,
        ) -> int:
            """
            """
            bmask_label = dialate_bmask(bmask_label, 3)
            intersection = bmask_input & bmask_label
            if not np.any(intersection):
                return

            labels_mask = (~bmask_input & bmask_label) & dialate_bmask(intersection, radius)
            labels, counts = np.unique(imask_input[labels_mask], return_counts=True)
            counts = counts[labels != 0] # ignore labels to be inpainted and dead labels
            labels = labels[labels != 0]
            if len(labels) == 0:
                return 0 # dead label that will result in occluded object not being activated from this frame
            imask_paint[intersection] = labels[np.argmax(counts)]

        def compute_inpaint_radius(
            bmask: NumpyTensor['h', 'w'], 
            imask: NumpyTensor['h', 'w'], ratio: float, clip_min=15, clip_max=75, bound_iterations=20, bound_threshold=5
        ) -> int:
            """
            """
            # assume bmask is a circle: r = sqrt(VOL / pi)(sqrt(c) - 1)
            assert ratio > 1
            radius = int(np.sqrt(np.sum(bmask) / np.pi * (ratio - 1)))
            radius = int(np.clip(radius, clip_min, clip_max))

            # ensure dialation doesn't overpaint non adjacent regions
            num_labels = len(np.unique(imask[dialate_bmask(bmask, clip_min)]))
            lo = 0
            hi = radius
            for _ in range(bound_iterations):
                mid = (lo + hi) // 2
                num_labels_current = len(np.unique(imask[dialate_bmask(bmask, mid)]))
                if num_labels_current > num_labels:
                    hi = mid
                else:
                    lo = mid
                if hi - lo < bound_threshold:
                    break
            radius = min(radius, lo)
            return radius

        def inpaint(index: int, downsample=2, radius=15):
            """
            """
            image = np.array(sequence.images[index])
            imask = np.array(sequence.imasks[index])
            
            bmask = np.zeros_like(sequence.imasks[index])
            for label in labels_to_inpaint:
                bmask_label = sequence.imasks[index] == label
                bmask_label = dialate_bmask(bmask_label, compute_inpaint_radius(bmask_label, imask, ratio=1.5))
                bmask |= bmask_label
            if not np.any(bmask):
                return
            
            imask[bmask] = -1 # remove labels to be inpainted
            imask = imask.astype(np.uint8)
            bmask = bmask.astype(np.uint8)
            #bmask = cv2.dilate(bmask, np.ones((radius, radius), np.uint8), iterations=1) # dialate to remove boundary artifacts

            # Downsample inputs for faster processing
            H, W = image.shape[:2]
            image_input = cv2.resize(image, (W // downsample, H // downsample))
            bmask_input = cv2.resize(bmask, (W // downsample, H // downsample), interpolation=cv2.INTER_NEAREST)
            imask_input = cv2.resize(imask, (W // downsample, H // downsample), interpolation=cv2.INTER_NEAREST)

            # Compute RGB inpainting and SAM masks
            image_paint = self.model_inpainter(image_input, bmask_input)
            bmasks_sam  = self.model_segmenter(image_paint, dialate=5)

            # Use SAM masks for instance mask inpainting
            imask_paint = np.array(imask_input)
            imask_paint = imask_paint.astype(int)
            bmask_input = bmask_input.astype(bool)
            for i, bmask_sam in enumerate(bmasks_sam):
                # predict separately for each sam bmask connected component for better locality
                for j, bmask_label in enumerate(connected_components(bmask_sam)):
                        predict_label(imask_input, bmask_input, imask_paint, bmask_label)
            
            # Upsample to original size
            image_paint = cv2.resize(image_paint, (W, H))
            imask_paint = cv2.resize(imask_paint, (W, H), interpolation=cv2.INTER_NEAREST)
            for label in labels_to_inpaint: # remove straggler labels
                imask_paint[imask_paint == label] = 0
            imask_paint = remove_artifacts_cmask(imask_paint, mode='holes'  , min_area=self.config.model_segmenter.min_area)
            imask_paint = remove_artifacts_cmask(imask_paint, mode='islands', min_area=self.config.model_segmenter.min_area)
            imask_paint = fill_background_holes(imask_paint, background=INSTANCE_BACKGROUND)
            
            # Write to sequence
            sequence_updated.images[index] = image_paint
            sequence_updated.imasks[index] = imask_paint
            colormap_image(sequence_updated.images[index]).save(f'tmp/image_paint_iter_{iteration}_index_{index}.png')
            colormap_cmask(sequence_updated.imasks[index]).save(f'tmp/imask_paint_iter_{iteration}_index_{index}.png')
            colormap_bmasks(bmasks_sam).save(f'tmp/bmasks_sam_iter_{iteration}_index_{index}.png')
            colormap_bmask(bmask).save(f'tmp/bmask_iter_{iteration}_index_{index}.png')

        for i in tqdm(range(len(sequence))):
            inpaint(i)
        return sequence_updated


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
        
        # Not enough memory to load all modules at once
        extractor = SequenceExtraction(self.config.extractor)
        inpainter = SequenceInpainting(self.config.inpaintor)

        images_total, sequences_total = defaultdict(dict), {0: sequence.clone()}
        for i in range(self.config.max_iterations):
            images = extractor(sequence, instance2semantic, iteration=i)
            print(colored(f'Iteration {i} extracted {len(images)} mesh image crops, now inpainting sequence', 'green'))
            if len(images) == 0:
                break
            sequence = inpainter(sequence, images.keys(), iteration=i)
            images_total[i], sequences_total[i] = images, sequence.clone()
        exit()

        # Create after extraction/inpainting to save GPU memory
        generator = ModelInstantMesh(self.config.generator)
        meshes_total = {}
        for label, image in images_total.items():
            print(colored(f'Converting label {label} to mesh', 'green'))
            mesh = generator(image)
            mesh.export(f'tmp/mesh_{label}.obj')
            meshes_total[label] = mesh
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
        'cache_dir': reader.save_dir / 'sequence_factorization_extraction'
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
        'cache_dir': reader.save_dir / 'sequence_factorization_inpainting'
    }
    generator_config = {
        'script_path': 'third_party/InstantMesh/run.py',
        'model_config_path': 'configs/instant-mesh-large.yaml',
        'image_path': 'examples/input.png',
        'tmesh_path': 'outputs/instant-mesh-large/meshes/input.obj'
    }
    sequence_factorization = SequenceFactorization(OmegaConf.create({
        'extractor': extractor_config,
        'inpaintor': inpainter_config,
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