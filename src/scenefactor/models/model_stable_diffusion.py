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
        self.config = config
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
    
    def process(
        self, prompt: str, image: NumpyTensor['h', 'w', 3], bmask: NumpyTensor['h', 'w']
    ) -> NumpyTensor['h', 'w', 3]:
        """
        """
        image = PIL.Image.fromarray(image).resize(self.RESIZE)
        bmask = PIL.Image.fromarray(bmask).resize(self.RESIZE)
        image_inpainted = self.pipeline(prompt=prompt, image=image, mask_image=bmask).images[0]
        image_inpainted = np.array(image_inpainted)
        return image_inpainted
    

if __name__ == '__main__':
    from diffusers.utils import load_image

    image_url = 'https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png'
    bmask_url = 'https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png'
    image = load_image(image_url).numpy()
    bmask = load_image(bmask_url).numpy()
    print(image.shape, bmask.shape)
    prompt = 'Face of a yellow cat, high resolution, sitting on a park bench'

    model = ModelStableDiffusion(OmegaConf.create({'checkpoint': 'benjamin-paine/stable-diffusion-v1-5-inpainting'}))
    image_inpainted = model(prompt, image, bmask)[0]
    print(image_inpainted.shape)

    PIL.fromarray(image).save('image.png')
    PIL.fromarray(bmask).save('bmask.png')
    PIL.fromarray(image_inpainted).save('image_inpainted.png')