import os
import itertools
from collections import defaultdict

import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from scenefactor.data.common import NumpyTensor
from scenefactor.data.sequence import FrameSequence
from scenefactor.models import ModelLama, ModelSam
from scenefactor.utils.geom import *
from scenefactor.utils.visualize import *
from scenefactor.factorization_common import *


EPSILON_RADIUS = 2


def order(x, y):
    return (min(x, y), max(x, y))


def build_border(shape):
    """
    """
    data = np.zeros(shape, dtype=np.uint8)
    data[ 0,  :] = 1
    data[-1,  :] = 1
    data[ :,  0] = 1
    data[ :, -1] = 1
    return data


class OcclusionResolver:
    """
    """
    def __init__(self, config: OmegaConf):
        """
        """
        self.config = config
        self.model_inpainter = ModelLama(config.inpainter)
        self.model_segmenter = ModelSam (config.segmenter)
        self.visualizations_path = self.config.cache / 'visualizations' if 'cache' in self.config else None

    def process_sequence(self, sequence: FrameSequence) -> list[dict]:
        """
        """    
        frames = []
        for i in tqdm(range(len(sequence)), desc='Resolving occlusions'):
            image = sequence.images[i]
            imask = sequence.imasks[i]

            if self.visualizations_path is not None:
                self.visualizations_path = self.visualizations_path / f'{i:03}'
                os.makedirs(self.visualizations_path, exist_ok=True)
                visualize_image(image).save(f'{self.visualizations_path}/image.png')
                visualize_cmask(imask).save(f'{self.visualizations_path}/imask.png')

            # compute labels that should not be processed
            BORDER = build_border(imask.shape)

            def intersect_border(label):
                return np.any(dialate_bmask(imask == label, radius=EPSILON_RADIUS) & BORDER)

            # obtain occulsion information for each pair of labels
            labels = defaultdict(dict)
            invalid = set()
            for label in np.unique(imask):
                if label == INSTANCE_BACKGROUND or np.sum(imask == label) < self.model_segmenter.config.min_area:
                    invalid.add(label)
                else:
                    labels[label]['adjacent'] = {}
            
            for label1, label2 in itertools.combinations(np.unique(imask), 2):
                if label1 in invalid or \
                   label2 in invalid:
                    continue
                data = self.resolve(image, imask, label1, label2)
                if data is None:
                    continue
                labels[label1]['adjacent'][label2] = data
                labels[label2]['adjacent'][label1] = {
                    'bmask1_delta': data['bmask2_delta'],
                    'bmask2_delta': data['bmask1_delta'],
                    'bmask1_ratio': data['bmask2_ratio'],
                    'bmask2_ratio': data['bmask1_ratio'],
                }

            # compute total occluded area/mask for each label
            for label1, info in labels.items():
                n = len(info['adjacent'])
                occluded_mask = np.zeros_like(imask)
                for label2, data in info['adjacent'].items():
                    occluded_mask |= data['bmask1_delta']
                info.update({
                    'occlusion_area': occluded_mask.sum(),
                    'occlusion_mask': occluded_mask,
                })
            
            # combine occlusion information to get per label metrics
            labels_processed = {}
            for label1, info in labels.items():
                occluded_cost = 0
                n = len(info['adjacent'])
                for label2, data in info['adjacent'].items():
                    ratio1 = data['bmask1_ratio']
                    ratio2 = data['bmask2_ratio']
                    if ratio1 < self.config.occlusion_threshold_ratio and \
                       ratio2 < self.config.occlusion_threshold_ratio:
                        continue
                    occluded_cost += min(ratio1 / ratio2, 10)
                occluded_cost = occluded_cost / n if n > 0 else 0
                labels_processed[label1] = info
                labels_processed[label1]['occlusion_cost'] = occluded_cost
                labels_processed[label1]['valid'] = not intersect_border(label1)
            
            # for k, v in labels_processed.items():
            #     print('--------------------------------------------')
            #     print('Label:', k)
            #     print(v['occlusion_cost'])
            #     print(v['occlusion_area'])
            #     print(v['valid'])
                
            frames.append(labels_processed)

            if self.visualizations_path is not None:
                self.visualizations_path = self.visualizations_path.parent
        return frames

    def resolve(
        self,
        image: NumpyTensor['h', 'w', 3],
        imask: NumpyTensor['h', 'w'],
        label1: int,
        label2: int,
    ) -> dict:
        """
        """
        if not np.any(
            dialate_bmask(imask == label1, radius=EPSILON_RADIUS) & # test if labels not adjacent at full resolution
            dialate_bmask(imask == label2, radius=EPSILON_RADIUS)
        ):
            return None
        
        H, W = imask.shape
        # downsample image and masks for faster processing
        image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))
        imask = cv2.resize(imask, (imask.shape[1] // 2, imask.shape[0] // 2), interpolation=cv2.INTER_NEAREST)
        
        bmask1 = imask == label1
        bmask2 = imask == label2
        if self.visualizations_path is not None:
            visualize_image(image) .save(f'{self.visualizations_path}/{label1}_{label2}_image.png')
            visualize_bmask(bmask1).save(f'{self.visualizations_path}/{label1}_{label2}_bmask1.png')
            visualize_bmask(bmask2).save(f'{self.visualizations_path}/{label1}_{label2}_bmask2.png')

        bmask1_fixed, image1_paint = self.repair(image, bmask1, bmask2)
        bmask2_fixed, image2_paint = self.repair(image, bmask2, bmask1)
        bmask1_delta = bmask1_fixed & bmask2
        bmask2_delta = bmask2_fixed & bmask1
        bmask1_ratio = np.sum(bmask1_delta) / np.sum(bmask2)
        bmask2_ratio = np.sum(bmask2_delta) / np.sum(bmask1)

        if self.visualizations_path is not None:
            visualize_bmask(bmask1_fixed).save(f'{self.visualizations_path}/{label1}_{label2}_bmask1_fixed.png')
            visualize_image(image1_paint).save(f'{self.visualizations_path}/{label1}_{label2}_image1_paint.png')
            visualize_bmask(bmask2_fixed).save(f'{self.visualizations_path}/{label1}_{label2}_bmask2_fixed.png')
            visualize_image(image2_paint).save(f'{self.visualizations_path}/{label1}_{label2}_image2_paint.png')

        return {
            'bmask1_delta': resize_bmask(bmask1_delta, (W, H)),
            'bmask2_delta': resize_bmask(bmask2_delta, (W, H)),
            'bmask1_ratio': bmask1_ratio,
            'bmask2_ratio': bmask2_ratio,
        }
    
    def repair(
        self,
        image : NumpyTensor['h', 'w', 3],
        bmask1: NumpyTensor['h', 'w'],
        bmask2: NumpyTensor['h', 'w'],
    ) -> tuple:
        """
        Complete bmask1 after inpainting bmask2.
        """
        sbmask = erode_bmask(bmask1, radius=self.config.segmenter_erosion)
        # thinner areas representation boosted via prunning instead of random sampling
        points = bmask_sample_points_grid(sbmask, 10, std=2)
        points = np.random.permutation(points)[:self.config.segmenter_num_samples]
        labels = np.ones(len(points))
        prompt = {
            'point_coords': points,
            'point_labels': labels, 'multimask_output': False
        }
        image_paint = self.model_inpainter(
            image, bmask2, dialate=self.config.inpainter_dialation, iterations=self.config.inpainter_iterations)
        bmask_fixed = self.model_segmenter(image_paint, prompt=prompt)
        if bmask_fixed is None:
            bmask_fixed = bmask1 # TODO hack for edge case
        else:
            bmask_fixed = bmask_fixed[0]

        # further refine bmask_fixed
        components = [c for c in connected_components(bmask_fixed) if np.any(c & bmask1)]
        bmask_fixed = np.zeros_like(bmask_fixed)
        for c in components:
            bmask_fixed |= c
        bmask_fixed |= bmask1 # sometimes overerosion

        return bmask_fixed, image_paint


if __name__ == '__main__':
    import shutil
    import os
    from pathlib import Path
    if Path('tmp').exists():
        shutil.rmtree('tmp')
    os.makedirs('tmp')

    from scenefactor.data.sequence_reader_replica_vmap import ReplicaVMapFrameSequenceReader
    reader = ReplicaVMapFrameSequenceReader(base_dir='/home/gtangg12/data/replica-vmap', name='room_0')
    sequence = reader.read(slice=(0, 100, 250))

    # from scenefactor.data.sequence_reader_graspnet import GraspNetFrameSequenceReader
    # reader = GraspNetFrameSequenceReader(base_dir='/home/gtangg12/data/graspnet', name='scene_0000')
    # sequence = reader.read(slice=(0, 200, 25))

    config = OmegaConf.load('configs/factorization.yaml')
    config.occlusion.cache = Path('tmp')

    import time
    start_time = time.time()
    occlusion_resolver = OcclusionResolver(config.occlusion)
    occlusions = occlusion_resolver.process_sequence(sequence)
    run_time = time.time() - start_time
    print('Elapsed time:', run_time)
    print('Mean time per iteration', run_time / len(sequence))