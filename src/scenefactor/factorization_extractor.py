from collections import defaultdict
from pathlib import Path

from omegaconf import OmegaConf
from termcolor import colored
from tqdm import tqdm

from scenefactor.data.common import NumpyTensor
from scenefactor.data.sequence import FrameSequence
from scenefactor.models import ModelClip
from scenefactor.utils.geom import *
from scenefactor.utils.visualize import *
from scenefactor.factorization_common import *


SequenceBmasks = dict[int, dict[int, NumpyTensor['h', 'w']]] # label: index: bmask
SequenceBboxes = dict[int, dict[int, BBox]]                  # label: index: bbox


def compute_sequence_bmasks_bboxes(sequence: FrameSequence, instance2semantic: dict, min_area=1024, background=0) -> tuple[
    SequenceBmasks,
    SequenceBboxes
]:
    """
    TODO: parallelize
    """
    bmasks = defaultdict(dict)
    bboxes = defaultdict(dict)
    for index, imask in tqdm(enumerate(sequence.imasks), desc='Sequence extraction computing bmasks and bboxes'):
        for label in np.unique(imask):
            if semantic_class(label, sequence, instance2semantic) in [None, 'stuff']:
                continue
            bmask = imask == label
            bmask = remove_artifacts(bmask, mode='islands', min_area=min_area)
            bmask = remove_artifacts(bmask, mode='holes'  , min_area=min_area)
            if not np.any(bmask):
                continue
            bmasks[label][index] = bmask
            bboxes[label][index] = compute_bbox(bmask)
    return bmasks, bboxes


class SequenceExtractor:
    """
    """
    def __init__(self, config: OmegaConf):
        """
        """
        self.config = config
    
    def __call__(self, sequence: FrameSequence, instance2semantic: dict[int, int], iteration: int) -> dict[int, NumpyTensor['h', 'w', 3]]:
        """
        """
        sequence_bmasks, sequence_bboxes = compute_sequence_bmasks_bboxes(sequence, instance2semantic, min_area=128, background=0)

        def compute_view_score(label: int, index: int) -> float:
            """
            """
            bmask = sequence_bmasks[label][index]
            imask = sequence.imasks[index]
            image = sequence.images[index]
            bbox  = sequence_bboxes[label][index]
            bbox_expanded      = resize_bbox(sequence_bboxes[label][index], mult=self.config.score_bbox_expand_mult)
            bbox_expanded_test = resize_bbox(bbox_expanded                , mult=self.config.score_bbox_expand_mult) # check bbox should be larger than expanded bbox to avoid (literal) edge cases

            # Condition 1: valid bbox
            if not bbox_check_bounds(bbox_expanded_test, *bmask.shape): 
                return 0, None
            bbox_occupancy = bmask.sum() / bbox_area(bbox)
            if bbox_occupancy < self.config.score_bbox_occupancy_threshold:
                return 0, None
            if bbox_area(bbox) < self.config.score_bbox_min_area:
                return 0, None
            
            # Condition 2: no holes due to occlusions # TODO check if area inside hole has component outside (if so don't count as hole)
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

            image_view = crop(remove_background(image, bmask, background=255, outline_thickness=None), bbox_expanded)
            semantic_label = instance2semantic[label]
            semantic_label = sequence.metadata['semantic_info'][semantic_label]['name']
            print(f'Score: {bbox_occupancy:.2f}, Index: {index}, Semantic Label: {semantic_label}')

            if self.config.visualize:
                visualizations_path = f'{self.config.cache}/visualizations'
                visualizations_tail = f'label_{label}_iter_{iteration}_index_{index}_score_{bbox_occupancy:.2f}'
                visualize_image(image_view).save(f'{visualizations_path}/image_{visualizations_tail}.png')
            
            return bbox_occupancy, image_view

        def process(label: int) -> list:
            """
            """
            sequence_labels = sequence_bmasks.keys()
            if label == INSTANCE_BACKGROUND or label not in sequence_labels:
                return []
            outputs = []
            for index, _ in sequence_bmasks[label].items():
                score, image = compute_view_score(label, index)
                if score > self.config.score_threshold:
                    outputs.append((image, score, index))
            outputs = sorted(outputs, key=lambda x: x[1], reverse=True)[:self.config.score_topk]
            return outputs

        label2crops = {}
        labels = np.unique(sequence.imasks)
        for label in labels:
            crops = process(label)
            if len(crops):
                print(colored(f'Extracting label {label}', 'green'))
                label2crops[label] = crops
        return label2crops