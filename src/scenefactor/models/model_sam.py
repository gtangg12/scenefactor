import re
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from omegaconf import OmegaConf

'''
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
'''
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry

from scenefactor.data.common import NumpyTensor
from scenefactor.utils.geom import remove_artifacts_cmask, combine_bmasks, decompose_cmask
from scenefactor.utils.colormaps import colormap_bmasks


def rasterize_clean_masks(bmasks: NumpyTensor['n', 'h', 'w'], min_area=128) -> NumpyTensor['n h w']:
    """
    Rasterizes and cleans masks. Removes holes/islands post rasterization.
    """
    cmask = combine_bmasks(bmasks, sort=True)
    cmask = remove_artifacts_cmask(cmask, mode='holes'  , min_area=min_area)
    cmask = remove_artifacts_cmask(cmask, mode='islands', min_area=min_area)
    bmasks = decompose_cmask(cmask)
    return np.stack(bmasks)


class ModelSAM():
    """
    """
    def __init__(self, config: OmegaConf, device='cuda'):
        """
        """
        self.config = config
        self.device = device

        #self.model = build_sam2(self.config.model_config, self.config.checkpoint, device=device)
        match = re.search(r'vit_(l|tiny|h)', self.config.checkpoint)
        self.model = sam_model_registry[match.group(0)](checkpoint=self.config.checkpoint)
        self.model.to(device)
        self.model.eval()
        self.engine = {
            'pred': SamPredictor,              #SAM2ImagePredictor,
            'auto': SamAutomaticMaskGenerator, #SAM2AutomaticMaskGenerator,
        }[self.config.mode](self.model, **self.config.get('engine_config', {}))
    
    def __call__(
        self, image: NumpyTensor['h', 'w', 3], prompt: dict = None, dilate: tuple[int, int] = None
    ) -> NumpyTensor['n', 'h', 'w']:
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
        masks = rasterize_clean_masks(masks, min_area=self.config.get('min_area', 128))
        if dilate is not None:
            masks = [
                cv2.dilate(mask.astype(np.uint8), dilate, iterations=1).astype(bool)
                for mask in masks
            ]
        return np.stack(masks)


if __name__ == '__main__':
    model = ModelSAM(OmegaConf.create({
        'checkpoint': '/home/gtangg12/scenefactor/checkpoints/sam_vit_h_4b8939.pth', #'/home/gtangg12/scenefactor/checkpoints/sam2_hiera_large.pt', 
        'model_config': 'sam2_hiera_l.yaml',
        'mode': 'auto',
        'engine_config': {}
    }))
    image = Image.open('tests/room.png').convert('RGB')
    image = np.asarray(image)
    masks = model(image)
    print(masks.shape)
    colormap_bmasks(masks, image).save('tests/sam_cmask.png')