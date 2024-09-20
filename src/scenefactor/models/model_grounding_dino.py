import cv2
import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf

from groundingdino.util.inference import load_model, predict

from scenefactor.data.common import NumpyTensor
from scenefactor.utils.geom import BBox, deduplicate_bboxes


class ModelGroundingDino:
    """
    """
    RESIZE_HEIGHT = 384

    def __init__(self, config: OmegaConf, device='cuda'):
        """
        """
        self.config = config
        self.device = device
        self.model = load_model(config.checkpoint_config, config.checkpoint)
    
    def __call__(self, image: NumpyTensor['h', 'w', 3], labels: list[str]) -> list[BBox]:
        """
        """
        H, W, _ = image.shape
        image = cv2.resize(image, (
            self.RESIZE_HEIGHT * image.shape[1] // image.shape[0], 
            self.RESIZE_HEIGHT
        ))
        image = torch.tensor(image).permute(2, 0, 1).to(self.device).float() / 255.
        
        bboxes = []
        logits = []
        for text in labels:
            bbox, logit, _ = predict(
                model=self.model,
                image=image,
                caption=text,
                box_threshold =self.config.bbox_threshold,
                text_threshold=self.config.text_threshold,
            )
            bbox  = bbox .cpu().numpy()
            logit = logit.cpu().numpy()
            bbox = (bbox * np.array([W, H, W, H])).astype(int)
            bbox = np.stack([
                bbox[:, 1] - bbox[:, 3] // 2, # ccwh -> tlbr
                bbox[:, 0] - bbox[:, 2] // 2,
                bbox[:, 1] + bbox[:, 3] // 2, 
                bbox[:, 0] + bbox[:, 2] // 2,
            ], axis=1)
            bboxes.append(bbox)
            logits.append(logit)
        bboxes = np.concatenate(bboxes, axis=0)
        logits = np.concatenate(logits, axis=0)

        sorted_indices = np.argsort(logits)[::-1]
        logits = logits[sorted_indices]
        bboxes = bboxes[sorted_indices]
        if len(bboxes) > 0:
            bboxes, indices = deduplicate_bboxes(bboxes, return_indices=True)
            logits = logits[indices]
        return bboxes.astype(int), logits


if __name__ == '__main__':
    image = Image.open('/home/gtangg12/scenefactor/tests/test.png').convert('RGB')
    image = np.array(image)
    model = ModelGroundingDino(OmegaConf.create({
        'checkpoint'       : '/home/gtangg12/scenefactor/checkpoints/groundingdino_swinb_cogcoor.pth',
        'checkpoint_config': '/home/gtangg12/scenefactor/third_party/GroundingDino/groundingdino/config/GroundingDINO_SwinB_cfg.py',
        'bbox_threshold': 0.25,
        'text_threshold': 0.25,
    }))
    bboxes, logits = model(image, ['dog', 'table', 'plant'])

    from scenefactor.utils.colormaps import colormap_bboxes
    colormap_bboxes(bboxes, image).save('/home/gtangg12/scenefactor/tests/test_grounding_dino.png')
