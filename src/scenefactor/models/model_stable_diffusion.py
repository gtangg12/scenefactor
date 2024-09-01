import PIL
import PIL.Image
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid

from scenefactor.data.common import NumpyTensor


class ModelStableDiffusion:
    """
    """
    RESIZE = (512, 512)

    def __init__(self, config: OmegaConf, device='cuda'):
        """
        """
        self.pipeline = AutoPipelineForInpainting.from_pretrained(config.checkpoint, torch_dtype=torch.float16)
        self.pipeline = self.pipeline.to(self.device)
    
    def __call__(
        self, prompt: str, image: NumpyTensor['n', 'h', 'w', 3], bmask: NumpyTensor['n', 'h', 'w']
    ) -> NumpyTensor['n', 'h', 'w', 3]:
        """
        """
        outputs = []
        for i in range(len(image)):
            image_inpainted = self.process(prompt, image[i], bmask[i])
            outputs.append(image_inpainted)
        return np.stack(outputs)
    
    def process(self, prompt: str, image: NumpyTensor['h', 'w', 3], bmask: NumpyTensor['h', 'w']) -> NumpyTensor['h', 'w', 3]:
        """
        """
        image = PIL.Image.fromarray(image).resize(self.RESIZE)
        bmask = PIL.Image.fromarray(bmask).resize(self.RESIZE)
        image_inpainted = self.pipeline(prompt=prompt, image=image, mask_image=bmask).images[0]
        image_inpainted = np.array(image_inpainted)
        return image_inpainted
    

if __name__ == '__main__':
    pass