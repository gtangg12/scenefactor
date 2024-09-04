import re
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from omegaconf import OmegaConf

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

from scenefactor.data.common import NumpyTensor
from scenefactor.utils.geom import combine_bmasks, decompose_mask


def remove_artifacts(mask: NumpyTensor['h w'], mode: str, min_area=128) -> NumpyTensor['h w']:
    """
    Removes small islands/fill holes from a mask.
    """
    assert mode in ['holes', 'islands']
    mode_holes = (mode == 'holes')

    def remove_helper(bmask):
        # opencv connected components operates on binary masks only
        bmask = (mode_holes ^ bmask).astype(np.uint8)
        nregions, regions, stats, _ = cv2.connectedComponentsWithStats(bmask, 8)
        sizes = stats[:, -1][1:]  # Row 0 corresponds to 0 pixels
        fill = [i + 1 for i, s in enumerate(sizes) if s < min_area] + [0]
        if not mode_holes:
            fill = [i for i in range(nregions) if i not in fill]
        return np.isin(regions, fill)

    mask_combined = np.zeros_like(mask)
    for label in np.unique(mask): # also process background
        mask_combined[remove_helper(mask == label)] = label
    return mask_combined


def colormap_mask(
    mask : NumpyTensor['h w'], 
    image: NumpyTensor['h w 3']=None, background=np.array([255, 255, 255]), foreground=None, blend=0.25
) -> Image.Image:
    """
    """
    palette = np.random.randint(0, 255, (np.max(mask) + 1, 3))
    palette[0] = background
    if foreground is not None:
        for i in range(1, len(palette)):
            palette[i] = foreground
    image_mask = palette[mask.astype(int)] # type conversion for boolean masks
    image_blend = image_mask if image is None else image_mask * (1 - blend) + image * blend
    image_blend = np.clip(image_blend, 0, 255).astype(np.uint8)
    return Image.fromarray(image_blend)


def colormap_bmask(bmask: NumpyTensor['h w']) -> Image.Image:
    """
    """
    return colormap_mask(bmask, background=np.array([0, 0, 0]), foreground=np.array([255, 255, 255]))


def colormap_bmasks(
    masks: NumpyTensor['n h w'], 
    image: NumpyTensor['h w 3']=None, background=np.array([255, 255, 255]), blend=0.25
) -> Image.Image:
    """
    """
    mask = combine_bmasks(masks)
    return colormap_mask(mask, image, background=background, blend=blend)


class ModelSAM2():
    """
    """
    def __init__(self, config: OmegaConf, device='cuda'):
        """
        """
        self.config = config
        self.device = device
        self.model = build_sam2(self.config.model_config, self.config.checkpoint, device=device)
        self.model.eval()
        self.engine = {
            'pred': SAM2ImagePredictor,
            'auto': SAM2AutomaticMaskGenerator,
        }[self.config.mode](self.model, **self.config.get('engine_config', {}))
    
    
    def __call__(self, image: NumpyTensor['h', 'w', 3], prompt: dict = None) -> NumpyTensor['n', 'h', 'w']:
        """
        For information on prompt format see:
        
        https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/predictor.py#L104
        """
        assert self.config.mode == ('auto' if prompt is None else 'pred')

        if self.config.mode == 'auto':
            annotations = self.engine.generate(image)
        else:
            self.engine.set_image(image)
            annotations = self.engine.predict(**prompt)[0]        
            annotations = [{'segmentation': m, 'area': m.sum().item()} for m in annotations] # Automatic Mask Generator format
        
        annotations = sorted(annotations, key=lambda x: x['area'], reverse=True)
        masks = np.stack([anno['segmentation'] for anno in annotations])
        return masks


if __name__ == '__main__':
    model = ModelSAM2(OmegaConf.create({
        'checkpoint': 'checkpoints/sam2_hiera_large.pt', 
        'model_config': 'sam2_hiera_l.yaml',
        'mode': 'auto'
    }))
    image = np.asarray(Image.open('tests/test.png').convert('RGB'))
    masks = model(image)
    print(masks.shape)
    colormap_bmasks(masks, image).save('tests/sam2_cmask.png')