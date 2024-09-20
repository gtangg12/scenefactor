import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf

from ram import inference_ram_openset as inference
from ram.models import ram_plus

from scenefactor.data.common import NumpyTensor
from scenefactor.models.transforms import transform_imagenet


class ModelRam:
    """
    """
    RESIZE = (384, 384)

    def __init__(self, config: OmegaConf, device='cuda'):
        """
        """
        self.config = config
        self.device = device

        self.transform = transform_imagenet(resize=self.RESIZE)
        self.model = ram_plus(pretrained=config.checkpoint, image_size=self.RESIZE[0], vit='swin_l')
        self.model.to(device)
        self.model.eval()

    def __call__(self, image: NumpyTensor['h', 'w', 3]) -> list[str]:
        """
        """
        return inference(self.transform(image).unsqueeze(0).to(self.device), self.model)


if __name__ == '__main__':
    image = Image.open('/home/gtangg12/scenefactor/tests/test.png').convert('RGB')
    image = np.array(image)
    model = ModelRam(OmegaConf.create({
        'checkpoint': '/home/gtangg12/scenefactor/checkpoints/ram_plus_swin_large_14m.pth'
    }))
    print(model(image))