import re
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.ops import masks_to_boxes
from PIL import Image
from omegaconf import OmegaConf

'''
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
'''
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry

from scenefactor.data.common import NumpyTensor
from scenefactor.models import ModelRam, ModelGroundingDino
from scenefactor.utils.geom import remove_artifacts_cmask, combine_bmasks, decompose_cmask, deduplicate_bmasks
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


class ModelSam:
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


class ModelSamGrounded:
    """
    """
    def __init__(self, config: OmegaConf, device='cuda'):
        """
        """
        self.config = config
        self.device = device

        self.model_ram = ModelRam(config.ram, device=device)
        self.model_grounding_dino = ModelGroundingDino(config.grounding_dino, device=device)
        self.model_sam_pred = ModelSam(config.sam_pred, device=device)
        self.model_sam_auto = ModelSam(config.sam_auto, device=device)

    def __call__(self, image: NumpyTensor['h', 'w', 3]) -> NumpyTensor['n', 'h', 'w']:
        """
        """
        labels = self.model_ram(image)['ram']
        bboxes, _ = self.model_grounding_dino(image, labels=[l.split(' | ') for l in labels])
        
        bmasks = []
        bboxes = []
        for bbox in bboxes:
            mask, bbox = self.model_sam_pred.process(image, prompt={'box': bbox.numpy()})
            bmasks.append(mask.permute(2, 0, 1))
            bboxes.append(bbox)
        bmasks, bboxes = zip(*sorted(zip(bmasks, bboxes), key=lambda x: x[0].sum().item(), reverse=True))
        bmasks = np.concatenate(bmasks) # (n, H, W)
        bboxes = np.concatenate(bboxes) # (n, 4)

        if self.config.include_sam_auto:
            bmasks_auto, bboxes_auto = self.model_sam_auto.process(image)
            bmasks_auto = bmasks_auto.permute(2, 0, 1) # (n, H, W)
            bmasks_auto = np.flip(bmasks_auto, axis=0) # sort SAM labels by increasing area
            bboxes_auto = np.flip(bboxes_auto, axis=0)
            # grounded sam takes precedence over sam auto
            bmasks = torch.concatenate([bmasks_auto, bmasks], axis=0) if len(bmasks) else bmasks_auto
            bboxes = torch.concatenate([bboxes_auto, bboxes], axis=0) if len(bboxes) else bboxes_auto

        bmasks, indices = deduplicate_bmasks(bmasks, return_indices=True)
        bboxes = bboxes[indices]
        bmasks = rasterize_clean_masks(bmasks, min_region_area=self.config.sam_pred.min_region_area)
        bboxes = masks_to_boxes(bmasks)
        return bmasks.astype(bool), bboxes.astype(int)


if __name__ == '__main__':
    model = ModelSam(OmegaConf.create({
        'checkpoint': '/home/gtangg12/scenefactor/checkpoints/sam_vit_h_4b8939.pth', #'/home/gtangg12/scenefactor/checkpoints/sam2_hiera_large.pt', 
        #'model_config': 'sam2_hiera_l.yaml',
        'mode': 'auto',
        'engine_config': {}
    }))
    image = Image.open('tests/room.png').convert('RGB')
    image = np.asarray(image)
    masks = model(image)
    print(masks.shape)
    colormap_bmasks(masks, image).save('tests/sam_cmask.png')

    model = ModelSamGrounded(OmegaConf.create({
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
        'include_sam_auto': True
    }))
    masks = model(image)
    print(masks.shape)
    colormap_bmasks(masks, image).save('tests/sam_grounded_cmask.png')
