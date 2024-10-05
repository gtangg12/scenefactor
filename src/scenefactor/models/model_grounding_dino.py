from collections import defaultdict

import cv2
import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from tqdm import tqdm

from groundingdino.util.inference import load_model, predict

from scenefactor.data.common import NumpyTensor
from scenefactor.utils.geom import BBox, bbox_area, bbox_overlap


def deduplicate_by_label(bboxes: NumpyTensor['n', 4], logits: NumpyTensor['n'], labels: list[str]):
    """
    """
    label2boxes = defaultdict(list)
    for bbox, logit, label in zip(bboxes, logits, labels):
        label2boxes[label].append((bbox, logit))
    
    label2boxes_deduplicated = defaultdict(list)
    for label, pairs in label2boxes.items():
        pairs = sorted(pairs, key=lambda x: bbox_area(x[0]), reverse=True)
        bboxes_unique = []
        logits_unique = []
        for bbox, logit in pairs:
            if not any([bbox_overlap(bbox, bbox_unique) > 0.5 for bbox_unique in bboxes_unique]):
                bboxes_unique.append(bbox)
                logits_unique.append(logit)
        label2boxes_deduplicated[label] = (np.stack(bboxes_unique), np.array(logits_unique))

    bboxes_output = []
    logits_output = []
    labels_output = []
    for label, (bboxes, logits) in label2boxes_deduplicated.items():
        bboxes_output.append(bboxes)
        logits_output.append(logits)
        labels_output.extend([label] * len(bboxes))
    return np.concatenate(bboxes_output, axis=0), \
           np.concatenate(logits_output, axis=0), \
           labels_output


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
        self.model.to(device)
        self.model.eval()
    
    def __call__(self, image: NumpyTensor['h', 'w', 3], texts: list[str], label_deduplicate=True) -> list[BBox]:
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
        labels = []
        for text in texts:
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
            bbox = np.stack([ # ccwh -> tlbr
                bbox[:, 1] - bbox[:, 3] // 2,
                bbox[:, 0] - bbox[:, 2] // 2,
                bbox[:, 1] + bbox[:, 3] // 2, 
                bbox[:, 0] + bbox[:, 2] // 2,
            ], axis=1)
            bboxes.append(bbox)
            logits.append(logit)
            labels.extend([text] * len(bbox))
        bboxes = np.concatenate(bboxes, axis=0)
        logits = np.concatenate(logits, axis=0)

        if label_deduplicate:
            bboxes, logits, labels = deduplicate_by_label(bboxes, logits, labels)
        sorted_indices = np.argsort(logits)[::-1]
        logits = logits[sorted_indices]
        bboxes = bboxes[sorted_indices]
        labels = [labels[i] for i in sorted_indices]

        return bboxes.astype(int), logits, labels


if __name__ == '__main__':
    image = Image.open('/home/gtangg12/scenefactor/tests/room2.png').convert('RGB')
    image = np.array(image)
    model = ModelGroundingDino(OmegaConf.create({
        'checkpoint'       : '/home/gtangg12/scenefactor/checkpoints/groundingdino_swinb_cogcoor.pth',
        'checkpoint_config': '/home/gtangg12/scenefactor/third_party/GroundingDino/groundingdino/config/GroundingDINO_SwinB_cfg.py',
        'bbox_threshold': 0.3,
        'text_threshold': 0.3,
    }))
    #bboxes, logits, labels = model(image, ['large, wooden drawer with objects on top', 'white lampshade near wall', 'ottoman chair', 'brown tabletop', 'jar on top of wooden drawer', 'books', 'plate'])
    #bboxes, logits, labels = model(image, ['flower vase', 'white lampshade', 'sofa', 'pillow'])
    bboxes, logits, labels = model(image, ['drawer'])
    from scenefactor.utils.visualize import visualize_bboxes
    visualize_bboxes(bboxes, image).save('/home/gtangg12/scenefactor/tests/test_grounding_dino.png')
    print(labels)
