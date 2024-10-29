from collections import defaultdict
from omegaconf import OmegaConf
from tqdm import tqdm

from scenefactor.data.common import NumpyTensor
from scenefactor.data.sequence import FrameSequence
from scenefactor.utils.geom import *
from scenefactor.utils.visualize import *
from scenefactor.sequence_extractor_ranking_losses import *


class SequenceExtractor:
    """
    """
    def __init__(self, config: OmegaConf):
        """
        """
        self.config = config
        loss_config = config.get('loss', {})
        self.module_loss_occ = OccupancyRankingLoss(loss_config.get('loss_occ', {}))
        self.module_loss_vlm = VLMRankingLoss      (loss_config.get('loss_vlm', {}))
        self.visualizations_path = f'{self.config.cache}/visualizations' if 'cache' in self.config else None

    def process_sequence(self, sequence: FrameSequence, frames: list[dict]) -> dict[int, NumpyTensor['h', 'w', 3]]:
        """
        """
        # Aggregate views for each label
        views = defaultdict(list)
        for image, imask, labels in zip(sequence.images, sequence.imasks, frames):
            for label, info in labels.items():
                if info['valid'] and info['occlusion_cost'] < self.config.occlusion_threshold_cost:
                    views[label].append(self.process(image, imask, label, info))
        if self.visualizations_path is not None:
            for label, view in views.items():
                for info in view:
                    index = info['index']
                    visualize_image(info['view']).save(f'{self.visualizations_path}/label_{label}_index_{index}.png')

        # Rank and extract the best view for each label
        outputs = {}
        for label, info in tqdm(views.items(), desc='Ranking views'):
            outputs[label] = self.select_view(info)
        if self.visualizations_path is not None:
            for label, image in outputs.items():
                visualize_image(image).save(f'{self.visualizations_path}/label_{label}_final.png')
        return outputs
    
    def process(
        self,
        image: NumpyTensor['h', 'w', 3],
        imask: NumpyTensor['h', 'w'],
        label: int, info: dict
    ):
        """
        """
        bmask = imask == label
        bbox = compute_bbox(bmask)
        view = crop(remove_background(image, bmask), bbox) # InstantMesh rescales border
        return {'bmask': bmask, 'bbox': bbox, 'view': view, **info}
    
    def select_view(self, views: dict) -> NumpyTensor['h', 'w', 3]:
        """
        """
        ocosts = [self.module_loss_occ(view) for view in views]
        images = [view['view'] for view in views]
        vindex = self.module_loss_vlm(images)
        return images[vindex] if vindex is not None else images[ocosts.index(min(ocosts))]