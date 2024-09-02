import cv2
import kornia
import matplotlib.pyplot as plt
import torch
from kornia.feature import LoFTR
from kornia_moons.viz import draw_LAF_matches
from omegaconf import OmegaConf
from PIL import Image

from scenefactor.data.common import NumpyTensor


class ModelLoFTR:
    """
    """
    RESIZE = (600, 375)

    def __init__(self, config: OmegaConf, device='cuda'):
        """
        """
        self.config = config
        self.device = device
        self.matcher = LoFTR(pretrained='indoor')
        self.matcher.to(self.device)

    def __call__(
        self, 
        images1: NumpyTensor['n', 'h', 'w', 3], 
        images2: NumpyTensor['n', 'h', 'w', 3],
        confidence_threshold: float = 0.9,
    ) -> list[dict]:
        """
        """
        images1 = torch.from_numpy(images1.permute(0, 3, 1, 2))
        images2 = torch.from_numpy(images2.permute(0, 3, 1, 2))
        images1 = kornia.geometry.resize(images1, self.RESIZE, antialias=True)
        images2 = kornia.geometry.resize(images2, self.RESIZE, antialias=True)
        inputs = {
            'image0': kornia.color.rgb_to_grayscale(images1),
            'image1': kornia.color.rgb_to_grayscale(images2),
        }
        with torch.inference_mode():
            outputs_matcher = self.matcher(inputs)
        outputs = {}
        for k, tensor in outputs_matcher.items():
            outputs[k] = tensor[outputs_matcher['confidence'] > confidence_threshold].cpu().numpy()
        return outputs
    

def fundamental_matrix(keypoints1: NumpyTensor['n', 2], keypoints2: NumpyTensor['n', 2]):
    """
    """
    Fmat, inliers = cv2.findFundamentalMat(keypoints1, keypoints2, cv2.USAC_MAGSAC, 0.5, 0.999, 100000)
    return Fmat, inliers > 0


def visualize_matches(
    image1: NumpyTensor['h', 'w', 3], 
    image2: NumpyTensor['h', 'w', 3], 
    keypoints1: NumpyTensor['n', 2],
    keypoints2: NumpyTensor['n', 2],
    inliers: NumpyTensor['h', 'w'], inlier_color=(0.2, 1, 0.2), feature_color=(0.2, 0.5, 1)
) -> Image.Image:
    """
    """
    fg = plt.figure()
    ax = fg.add_subplot(1, 1, 1)

    draw_LAF_matches(
        kornia.feature.laf_from_center_scale_ori(
            torch.from_numpy(keypoints1)   .view(1, -1,  2),
            torch.ones(keypoints1.shape[0]).view(1, -1,  1,  1),
            torch.ones(keypoints1.shape[0]).view(1, -1,  1),
        ),
        kornia.feature.laf_from_center_scale_ori(
            torch.from_numpy(keypoints2)   .view(1, -1,  2),
            torch.ones(keypoints2.shape[0]).view(1, -1,  1,  1),
            torch.ones(keypoints2.shape[0]).view(1, -1,  1),
        ),
        torch.arange(keypoints1.shape[0]).view(-1, 1).repeat(1, 2),
        kornia.tensor_to_image(image1),
        kornia.tensor_to_image(image2),
        inliers,
        draw_dict={'inlier_color': inlier_color, 'tentative_color': None, 'feature_color': feature_color, 'vertical': False},
        ax=ax,
    )

    return Image.frombytes('RGB', fg.canvas.get_width_height(), fg.canvas.tostring_rgb())


if __name__ == '__main__':
    pass