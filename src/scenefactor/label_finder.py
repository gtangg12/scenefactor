import itertools

import igraph
import numpy as np
import torch
from omegaconf import OmegaConf

from scenefactor.data.common import NumpyTensor
from scenefactor.data.sequence import FrameSequence
from scenefactor.models import ModelLama, ModelSam
from scenefactor.utils.geom import *
from scenefactor.utils.visualize import *
from scenefactor.factorization_common import *

EPSILON_RADIUS = 2


class SequenceFactorizationLabelFinder:
    """
    """
    def __init__(self, config: OmegaConf):
        """
        """
        self.config = config
        self.model_inpainter = ModelLama(config.inpainter)
        self.model_segmenter = ModelSam (config.segmenter)

    def __call__(self, sequence: FrameSequence) -> list[list[int]]:
        """
        """    
        frames = []
        for i, (image, imask) in enumerate(zip(sequence.images, sequence.imasks)):
            #visualize_cmask(imask).save(f'tmp/imask_{i:003}.png')
            labels = []
            for label in np.unique(imask):
                if label == INSTANCE_BACKGROUND:
                    continue
                if not self.process(image, imask, label):
                    labels.append(label)
            frames.append(labels)
        return frames
    
    def process(
        self,
        image: NumpyTensor['h', 'w', 3],
        imask: NumpyTensor['h', 'w'],
        label: int
    ) -> bool:
        """
        Determine if label is suitable for extraction i.e. not occluded by other labels.
        """
        bmask = imask == label
        
        if bbox_check_bounds(resize_bbox(compute_bbox(bmask), mult=1.5), *bmask.shape):
            return False
        
        for label2 in np.unique(imask):
            if label2 == INSTANCE_BACKGROUND or label2 == label:
                continue
            if self.resolve(image, imask, label, label2) == 2:
                return False
        return True

    def resolve(
        self,
        image: NumpyTensor['h', 'w', 3],
        imask: NumpyTensor['h', 'w'],
        label1: int,
        label2: int,
    ) -> int:
        """
        Returns code denoting occlusion type between label1 and label2:
            - 0 if no occlusion
            - 1 if label1 occludes label2
            - 2 if label2 occludes label1.
        """
        if not np.any(
            dialate_bmask(imask == label1, radius=EPSILON_RADIUS) & # test if objects not adjacent at full resolution
            dialate_bmask(imask == label2, radius=EPSILON_RADIUS)
        ):
            return 0
        
        # downsample image and masks for faster processing
        image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))
        imask = cv2.resize(imask, (imask.shape[1] // 2, imask.shape[0] // 2), interpolation=cv2.INTER_NEAREST)
        
        bmask1 = imask == label1
        bmask2 = imask == label2
        #visualize_bmask(bmask1).save(f'tmp/{label1}_{label2}_bmask1.png')
        #visualize_bmask(bmask2).save(f'tmp/{label1}_{label2}_bmask2.png')
    
        swap = False
        if np.sum(bmask1) < np.sum(bmask2):
            bmask1, bmask2 = bmask2, bmask1
            swap = True
        bmask1_area = np.sum(bmask1)
        bmask2_area = np.sum(bmask2)

        bmask1_fixed = self.repair(image, bmask1, bmask2, label1, label2, swap, 1)
        label1_ratio = np.sum(bmask1_fixed & bmask2) / np.sum(bmask2)
        bmask2_fixed = self.repair(image, bmask2, bmask1, label1, label2, swap, 2)
        label2_ratio = np.sum(bmask2_fixed & bmask1) / np.sum(bmask1)

        if bmask1_area > self.config.occlusion_one_sided_mult * bmask2_area:
            if label1_ratio > self.config.occlusion_one_sided_threshold_ratio:
                return 2 if not swap else 1
            else:
                return 1 if not swap else 2

        if label1_ratio < self.config.occlusion_threshold_ratio and \
           label2_ratio < self.config.occlusion_threshold_ratio:
            return 0
        if label1_ratio > label2_ratio:
            return 2 if not swap else 1
        else:
            return 1 if not swap else 2
    
    def repair(
        self,
        image : NumpyTensor['h', 'w', 3],
        bmask1: NumpyTensor['h', 'w'],
        bmask2: NumpyTensor['h', 'w'],
        label1: int,
        label2: int, swap: bool, call: int
    ) -> tuple:
        """
        Complete bmask1 after inpainting bmask2.
        """
        points = bmask_sample_points(bmask1, self.config.segmenter_num_samples, indexing='xy')
        labels = np.ones(len(points))
        prompt = {
            'point_coords': points,
            'point_labels': labels, 'multimask_output': False
        }
        image_paint = self.model_inpainter(image, bmask2, dialate=0)
        bmask_fixed = self.model_segmenter(image_paint, prompt=prompt)
        if bmask_fixed is None:
            bmask_fixed = bmask1 # TODO hack for edge case
        else:
            bmask_fixed = bmask_fixed[0]
        label = call if not swap else (1 if call == 2 else 2)
        #visualize_image(image_paint).save(f'tmp/{label1}_{label2}_image{label}.png')
        #visualize_bmask(bmask_fixed).save(f'tmp/{label1}_{label2}_bmask{label}_fixed.png')
        #plot_points(visualize_image(image_paint), points).save(f'tmp/{label1}_{label2}_image{label}_points.png')
        #plot_points(visualize_bmask(bmask_fixed), points).save(f'tmp/{label1}_{label2}_bmask{label}_points_fixed.png')
        return bmask_fixed


class SequenceFactorizationExtractor:
    """
    Clip view optimization
    """
    def __init__(self, config: OmegaConf):
        """
        """
        self.config = config
    
    def __call__(self, sequence: FrameSequence, hull: list[list[int]], iteration: int) -> dict[int, NumpyTensor['h', 'w', 3]]:
        """
        """
        # train nerf on sequence
        # 1 refine inpainting, semantics, also get depth
        # load nerf
        # multi source camera optimization from init views in hull
        label2object_view = {}
        for label in hull:
            label2object_view[label] = self.process(sequence, label, iteration)
        return label2object_view
    
    def process(self):
        """
        """
        pass


class SequenceFactorizationInpainter:
    """
    """
    pass


class SequenceFactorizationGenerator:
    """
    """
    pass


from trimesh.base import Trimesh
from scenefactor.utils.geom import connected_components


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
        occlusion_resolver = SequenceFactorizationLabelFinder(self.config.label_finder)
        occlusions = occlusion_resolver(sequence)
        exit()

        labels = set(np.unique(sequence.imasks)) # active labels

        def extract_hull():
            """
            """
            hull = set()
            # compute elements in hull
            for label in labels:
                if any([label in g.vs and g.vs[label].outdegree() > 0 for g in occlusions]):
                    continue
                hull.add(label)
            # remove elements from occlusion graphs
            for label in hull:
                for g in occlusions:
                    if label in g.vs:
                        g.delete_vertices(label)
            # remove elements from active labels
            for label in hull:
                labels.remove(label)
            return hull

        torch.cuda.empty_cache()
        extractor = SequenceFactorizationExtractor(self.config.extractor)
        inpainter = SequenceFactorizationInpainter(self.config.inpainter)

        current_sequence = sequence.clone()
        accumlated_crops = []
        iteration = 0
        while True:
            hull = extract_hull()
            if len(hull) == 0:
                assert len(labels) == 0, 'All labels should be processed.'
                break
            label2objectcrop = extractor(current_sequence, hull, iteration)
            current_sequence = inpainter(current_sequence, label2objectcrop, iteration)
            accumlated_crops.extend(label2objectcrop.values())
            iteration += 1

        torch.cuda.empty_cache()
        generator = SequenceFactorizationGenerator(self.config.generator)
        meshes = generator(accumlated_crops)
        return meshes


if __name__ == '__main__':
    import shutil
    import os
    if Path('tmp').exists():
        shutil.rmtree('tmp')
    os.makedirs('tmp')

    from scenefactor.data.sequence_reader_replica_vmap import ReplicaVMapFrameSequenceReader

    reader = ReplicaVMapFrameSequenceReader(base_dir='/home/gtangg12/data/replica-vmap', name='room_0')
    sequence = reader.read(slice=(0, -1, 500))

    config = OmegaConf.load('configs/factorization.yaml')

    model = SequenceFactorization(config)
    meshes = model(sequence)