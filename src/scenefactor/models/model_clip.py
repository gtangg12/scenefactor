import numpy as np
import clip
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor
from PIL import Image
from omegaconf import OmegaConf

from scenefactor.data.common import NumpyTensor, TorchTensor
from scenefactor.models.transforms import transforms_imagenet


DEFAULT_NEGATIVES = ['object', 'things', 'stuff', 'texture']


def norm(x: TorchTensor[..., 'd']) -> TorchTensor[..., 'd']:
    """
    """
    return x / x.norm(dim=-1, keepdim=True)


class ModelClip:
    """
    """
    EMBEDDING_DIM = 512

    def __init__(self, config: OmegaConf, device='cuda'):
        """
        """
        self.config = config
        self.device = device
        self.model, _ = clip.load(config.name)
        self.transform = transforms_imagenet(resize=(224, 224))
    
    def __call__(self, image: NumpyTensor['h', 'w', 3], text: str, negatives: list[str] = DEFAULT_NEGATIVES) -> float:
        """
        Return similarity score between image and text.
        """
        image_embeddings = self.encode_image(to_tensor(image).unsqueeze(0))

        positive_embeddings = self.encode_text(text)
        negative_embeddings = self.encode_text(negatives)
        positive_similarity = image_embeddings @ positive_embeddings.T
        negative_similarity = image_embeddings @ negative_embeddings.T
        probs = F.softmax(torch.cat([
            positive_similarity / self.config.temperature, 
            negative_similarity / self.config.temperature,
        ], dim=1), dim=1)
        return probs[0, 0].item()

    def encode_image(self, image: TorchTensor['batch', 'channels', 'h', 'w']) -> TorchTensor['batch', 512]:
        """
        """
        image = self.transform(image).cuda()
        with torch.no_grad():
            embedding = self.model.encode_image(image)
        return norm(embedding)

    def encode_text(self, text: str | list[str]) -> TorchTensor['batch', 512]:
        """
        """
        if isinstance(text, str):
            text = [text]
        tokens = clip.tokenize(text).cuda()
        with torch.no_grad():
            embedding = self.model.encode_text(tokens)
        return norm(embedding)


if __name__ == '__main__':
    model = ModelClip(OmegaConf.create({'name': 'ViT-B/32', 'temperature': 0.1}))
    image = Image.open('/home/gtangg12/scenefactor/tests/lama_image.png')
    image = np.array(image)
    print(model(image, 'cat'))
    print(model(image, 'dog'))