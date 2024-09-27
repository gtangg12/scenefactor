import yaml
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from omegaconf import OmegaConf

from scenefactor.data.common import NumpyTensor

LAMA_PATH = Path(__file__).parents[3] / 'third_party/lama'
import sys
sys.path.append(str(LAMA_PATH))
from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.training.trainers import load_checkpoint as load_checkpoint
from saicinpainting.evaluation.data import pad_tensor_to_modulo
from saicinpainting.evaluation.refinement import refine_predict


def load_lama_checkpoint(checkpoint: Path | str, device='cuda', gpus='0') -> tuple[nn.Module, OmegaConf]:
    """
    Load Lama checkpoint, which is structured as directory instead of .pt file.
    """
    def load_config(path):
        with open(path, 'r') as f:
            config = OmegaConf.create(yaml.safe_load(f))
        return config
    
    checkpoint = Path(checkpoint)
    predict_config = load_config(LAMA_PATH / 'configs/prediction/default.yaml')
    predict_config.model.path = checkpoint
    predict_config.refiner.gpu_ids = gpus
    trainer_config = load_config(checkpoint / 'config.yaml')
    trainer_config.training_model.predict_only = True
    trainer_config.visualizer.kind = 'noop'

    checkpoint = checkpoint / 'models' / predict_config.model.checkpoint
    model = load_checkpoint(trainer_config, checkpoint, strict=False, map_location='cpu')
    model.freeze()
    if not predict_config.get('refine', False):
        model.to(device)
    return model, predict_config


class ModelLama:
    """
    """
    def __init__(self, config: OmegaConf, device='cuda'):
        """
        """
        self.config = config
        self.device = device
        self.model, self.model_config = load_lama_checkpoint(config.checkpoint, device)
    
    def __call__(
        self, 
        image: NumpyTensor['h', 'w', 3], 
        bmask: NumpyTensor['h', 'w'], refine=False
    ) -> NumpyTensor['h', 'w', 3]:
        """
        """
        H, W = image.shape[:2]
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0) / 255
        bmask = torch.from_numpy(bmask)[None, None]
        image = pad_tensor_to_modulo(image, mod=8)
        bmask = pad_tensor_to_modulo(bmask, mod=8)
        
        batch = {'image': image.float(), 'mask': bmask, 'unpad_to_size': (torch.tensor([H]), torch.tensor([W]))}
        batch = move_to_device(batch, self.device)
        if refine:
            output = refine_predict(batch, self.model, **self.model_config.refiner)
        else:
            output = self.model(batch)[self.model_config.out_key]
        output = output[0].permute(1, 2, 0).detach().cpu().numpy()
        output = output[:H, :W]
        output = np.clip(output * 255, 0, 255).astype(np.uint8)
        return output


if __name__ == '__main__':
    from diffusers.utils import load_image

    image_url = 'https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png'
    bmask_url = 'https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png'
    image = np.array(load_image(image_url))
    bmask = np.array(load_image(bmask_url).convert('L'))
    model = ModelLama(OmegaConf.create({
        'checkpoint': '/home/gtangg12/scenefactor/checkpoints/big-lama'
    }))
    image_inpainted = model(image, bmask)
    print(image_inpainted.shape)

    Image.fromarray(image).save('tests/lama_image.png')
    Image.fromarray(bmask).save('tests/lama_bmask.png')
    Image.fromarray(image_inpainted).save('tests/lama_image_inpainted.png')

    image_inpainted = model(image, bmask, refine=True)
    Image.fromarray(image_inpainted).save('tests/lama_image_inpainted_refined.png')