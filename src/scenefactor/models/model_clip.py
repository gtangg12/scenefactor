from typing import Literal

import numpy as np
import clip
import torch
from PIL import Image

from scenefactor.data.common import NumpyTensor, TorchTensor
from scenefactor.models.transforms import transform_imagenet


class Clip:
    """
    """
    def __init__(self, name: Literal['ViT-B/32', 'ViT-B/16'] = 'ViT-B/32'):
        """
        """
        assert torch.cuda.is_available(), 'Clip requires CUDA'
        super().__init__()
        self.model, self.preprocess = clip.load(name)
    
    def __call__(self, image: NumpyTensor['h', 'w', 3], text: str) -> float:
        """
        Return similarity score between image and text.
        """
        norm = lambda x: x / x.norm(dim=-1, keepdim=True)

        image = self.preprocess(Image.fromarray(image)).unsqueeze(0).cuda()
        query = clip.tokenize([text]).cuda()
        with torch.no_grad():
            image_embeddings = self.model.encode_image(image)
            query_embeddings = self.model.encode_text (query)
        image_embeddings = norm(image_embeddings)
        query_embeddings = norm(query_embeddings)
        return (image_embeddings @ query_embeddings.T).item()


if __name__ == '__main__':
    model = Clip('ViT-B/16')
    image = Image.open('/home/gtangg12/scenefactor/tests/lama_image.png')
    image = np.array(image)
    print(model(image, 'cat'))
    print(model(image, 'dog'))