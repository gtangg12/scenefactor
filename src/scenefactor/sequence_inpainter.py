import os

from omegaconf import OmegaConf
from tqdm import tqdm

from scenefactor.data.common import NumpyTensor
from scenefactor.data.sequence import FrameSequence
from scenefactor.models import ModelLama, ModelSam, ModelSamGrounded
from scenefactor.utils.geom import *
from scenefactor.utils.visualize import *
from scenefactor.factorization_common import *


class SequenceInpainter:
    """
    """
    def __init__(self, config: OmegaConf):
        """
        """
        self.config = config
        self.model_inpainter = ModelLama(config.model_inpainter)
        self.model_segmenter = ModelSamGrounded(config.model_segmenter)
        self.visualizations_path = f'{self.config.cache}/visualizations' if 'cache' in self.config else None

    def process_sequence(self, sequence: FrameSequence, labels_to_inpaint: set[int]) -> FrameSequence:
        """
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
            counts = counts[labels != -1] # ignore dead labels
            labels = labels[labels != -1]
            if len(labels) == 0:
                return -1 # dead label that will result in occluded object not being activated from this frame
            imask_paint[intersection] = labels[np.argmax(counts)]

        def inpaint(index: int, downsample=2):
            """
            """
            image = np.array(sequence.images[index])
            imask = np.array(sequence.imasks[index])
            
            bmask = np.zeros_like(imask)
            for label in labels_to_inpaint:
                bmask_label = imask == label
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
            bmask_model = self.model_segmenter(image_paint, dialate=5)

            # Use SAM masks for instance mask inpainting
            imask_paint = np.array(imask_input)
            imask_paint = imask_paint.astype(int)
            bmask_input = bmask_input.astype(bool)
            for i, bmask_label in enumerate(bmask_model):
                # predict separately for each sam bmask connected component for better locality
                for j, bmask_label_component in enumerate(connected_components(bmask_label)):
                        predict_label(imask_input, bmask_input, imask_paint, bmask_label_component)
            
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
            
            if self.visualizations_path is not None:
                visualize_image (image)      .save(f'{self.visualizations_path}/index_{index}_image.png')
                visualize_image (image_paint).save(f'{self.visualizations_path}/index_{index}_image_paint.png')
                visualize_cmask (imask_paint).save(f'{self.visualizations_path}/index_{index}_imask_paint.png')
                visualize_bmask (bmask_input).save(f'{self.visualizations_path}/index_{index}_bmask_input.png')
                visualize_bmasks(bmask_model).save(f'{self.visualizations_path}/index_{index}_bmask_model.png')

        for i in tqdm(range(len(sequence))):
            inpaint(i)
        return sequence_updated