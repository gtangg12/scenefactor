import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from diffusers import StableDiffusionInpaintPipeline
from diffusers.utils import load_image

from scenefactor.data.common import NumpyTensor


class ModelStableDiffusion:
    """
    """
    RESIZE = (512, 512)

    def __init__(self, config: OmegaConf, device='cuda'):
        """
        """
        self.config = config
        self.device = device
        self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            config.checkpoint, torch_dtype=torch.float16, variant='fp16'
        )
        self.pipeline = self.pipeline.to(device)
    
    def __call__(self, prompt: str, image: NumpyTensor['h', 'w', 3], bmask: NumpyTensor['h', 'w']) -> NumpyTensor['h', 'w', 3]:
        """
        """
        image_size = image.shape[:2]
        image = Image.fromarray(image).resize(self.RESIZE)
        bmask = Image.fromarray(bmask).resize(self.RESIZE)
        image_inpainted = self.pipeline(prompt=prompt, image=image, mask_image=bmask).images[0]
        image_inpainted = image_inpainted.resize((image_size[1], image_size[0]))
        image_inpainted = np.array(image_inpainted)
        return image_inpainted
    

if __name__ == '__main__':
    from diffusers.utils import load_image

    image_url = 'https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png'
    bmask_url = 'https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png'
    image = np.asarray(load_image(image_url))
    bmask = np.asarray(load_image(bmask_url).convert('L'))
    #prompt = 'Face of a yellow cat, high resolution, sitting on a park bench'
    prompt = 'Fill in the missing parts of the image'

    model = ModelStableDiffusion(OmegaConf.create({'checkpoint': 'benjamin-paine/stable-diffusion-v1-5-inpainting'}))
    image_inpainted = model(prompt, image, bmask)
    print(image_inpainted.shape)

    Image.fromarray(image).save('tests/stable_diffusion_image.png')
    Image.fromarray(bmask).save('tests/stable_diffusion_bmask.png')
    Image.fromarray(image_inpainted).save('tests/stable_diffusion_image_inpainted.png')