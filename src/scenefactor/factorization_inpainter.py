from omegaconf import OmegaConf
from tqdm import tqdm

from scenefactor.data.common import NumpyTensor
from scenefactor.data.sequence import FrameSequence
from scenefactor.models import ModelLama, ModelSamGrounded
from scenefactor.utils.geom import *
from scenefactor.utils.colormaps import *
from scenefactor.factorization_utils import *


class SequenceInpainter:
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