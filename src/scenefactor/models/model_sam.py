import re
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from omegaconf import OmegaConf
from tqdm import tqdm

'''
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
'''
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry

from scenefactor.data.common import NumpyTensor
from scenefactor.models import ModelRam, ModelGroundingDino
from scenefactor.utils.geom import *
from scenefactor.utils.colormaps import *


def rasterize_clean_masks(bmasks: NumpyTensor['n', 'h', 'w'], min_area=128) -> NumpyTensor['n', 'h', 'w'] | None:
    """
    Rasterizes and cleans masks. Removes holes/islands post rasterization.
    """
    cmask = combine_bmasks(bmasks)
    cmask = remove_artifacts_cmask(cmask, mode='holes'  , min_area=min_area)
    cmask = remove_artifacts_cmask(cmask, mode='islands', min_area=min_area)
    bmasks = decompose_cmask(cmask, background=0) # combine_bmasks denotes background as 0
    if len(bmasks) == 0:
        return None
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
    
    def __call__(self, image: NumpyTensor['h', 'w', 3], prompt: dict=None, same_image=False) -> NumpyTensor['n', 'h', 'w']:
        """
        For information on prompt format see:
        
        https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/predictor.py#L104
        """
        assert self.config.mode == ('auto' if prompt is None else 'pred')
        
        if self.config.mode == 'auto':
            annotations = self.engine.generate(image)
        else:
            if not same_image:
                self.engine.set_image(image)
            annotations = self.engine.predict(**prompt)[0]
            annotations = [{'segmentation': m, 'area': m.sum().item()} for m in annotations] # Automatic Mask Generator format
        annotations = sorted(annotations, key=lambda x: x['area'], reverse=True)
        
        bmasks = np.stack([anno['segmentation'] for anno in annotations])
        bmasks = rasterize_clean_masks(bmasks, min_area=self.config.get('min_area', 128))
        return np.stack(bmasks) if bmasks is not None else None


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

    def __call__(self, image: NumpyTensor['h', 'w', 3], return_bboxes=False) -> NumpyTensor['n', 'h', 'w']:
        """
        """
        labels = self.model_ram(image)
        prompt_bboxes, _, _ = self.model_grounding_dino(image, labels.split(' | '))

        bmasks = []
        for i, bbox in enumerate(prompt_bboxes):
            prompt = {'box': bbox[[1, 0, 3, 2]], 'multimask_output': False} # tlbr -> xyxy
            mask = self.model_sam_pred(image, prompt, same_image=(i > 0))
            if mask is None:
                continue
            bmasks.append(mask[0])
        bmasks = sorted(bmasks, key=lambda x: x[0].sum().item(), reverse=True)
        bmasks = np.stack(bmasks) # (n, H, W)

        if self.config.include_sam_auto:
            bmasks_auto = self.model_sam_auto(image)
            bmasks_auto = np.flip(bmasks_auto, axis=0) # sort SAM labels by increasing area
            # grounded sam takes precedence over sam auto
            bmasks = np.concatenate([bmasks_auto, bmasks], axis=0) if len(bmasks) else bmasks_auto

        bmasks = deduplicate_bmasks(bmasks)
        bmasks = rasterize_clean_masks(bmasks, min_area=self.config.get('min_area', 128))
        bboxes = torchvision.ops.masks_to_boxes(torch.from_numpy(bmasks)).numpy()
        bboxes = bboxes[:, [1, 0, 3, 2]] # xyxy -> tlbr
        bmasks = bmasks.astype(bool)
        bboxes = bboxes.astype(int)
        if return_bboxes:
            return bmasks, bboxes
        return bmasks


if __name__ == '__main__':
    image = Image.open('tests/room3.png').convert('RGB')
    image = np.asarray(image)
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
        'include_sam_auto': True,
        'min_area': 256,
    }))
    bmasks, bboxes = model(image, return_bboxes=True)
    print(bmasks.shape)
    colormap_bmasks(bmasks, image).save('tests/sam_grounded_cmask.png')
    colormap_bboxes(bboxes, image).save('tests/sam_grounded_bboxes.png')