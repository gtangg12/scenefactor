import matplotlib.pyplot as plt
import cv2
import kornia
import numpy as np
import torch
from kornia.feature import LoFTR
from kornia_moons.viz import draw_LAF_matches
from omegaconf import OmegaConf
from PIL import Image

from scenefactor.data.common import NumpyTensor


def fundamental_matrix(
    keypoints0: NumpyTensor['n', 2], 
    keypoints1: NumpyTensor['n', 2]
) -> tuple[NumpyTensor['3', '3'], NumpyTensor['n']]:
    """
    """
    fmat, inliers = cv2.findFundamentalMat(keypoints0, keypoints1, cv2.USAC_MAGSAC, 0.5, 0.999, 100000)
    return fmat, inliers > 0


def visualize_matches(
    image0: NumpyTensor['h', 'w', 3], 
    image1: NumpyTensor['h', 'w', 3], 
    keypoints0: NumpyTensor['n', 2],
    keypoints1: NumpyTensor['n', 2],
    inliers: NumpyTensor['h', 'w'], inlier_color=(0.2, 1, 0.2), feature_color=(0.2, 0.5, 1)
) -> Image.Image:
    """
    """
    fg = plt.figure()
    ax = fg.add_subplot(1, 1, 1)

    draw_LAF_matches(
        kornia.feature.laf_from_center_scale_ori(
            torch.from_numpy(keypoints0)   .view(1, -1,  2),
            torch.ones(keypoints0.shape[0]).view(1, -1,  1,  1),
            torch.ones(keypoints0.shape[0]).view(1, -1,  1),
        ),
        kornia.feature.laf_from_center_scale_ori(
            torch.from_numpy(keypoints1)   .view(1, -1,  2),
            torch.ones(keypoints1.shape[0]).view(1, -1,  1,  1),
            torch.ones(keypoints1.shape[0]).view(1, -1,  1),
        ),
        torch.arange(keypoints0.shape[0]).view(-1, 1).repeat(1, 2),
        image0, image1, inliers,
        draw_dict={'inlier_color': inlier_color, 'tentative_color': None, 'feature_color': feature_color, 'vertical': False},
        ax=ax,
    )

    fg.canvas.draw()
    return Image.frombytes('RGB', fg.canvas.get_width_height(), fg.canvas.tostring_rgb())


class ModelLoFTR:
    """
    """
    RESIZE = (600, 375)

    def __init__(self, config: OmegaConf, device='cuda'):
        """
        """
        assert config.mode in ['outdoor', 'indoor']
        self.config = config
        self.device = device
        self.matcher = LoFTR(pretrained=self.config.mode)
        self.matcher.to(self.device)

    def __call__(
        self, 
        images1: NumpyTensor['h', 'w', 3], 
        images2: NumpyTensor['h', 'w', 3], confidence_threshold=0.75, ransac=True, visualize=True,
    ) -> list[dict]:
        """
        """
        images1 = torch.from_numpy(images1).permute(2, 0, 1).unsqueeze(0) / 255 # uint8 to float32
        images2 = torch.from_numpy(images2).permute(2, 0, 1).unsqueeze(0) / 255
        images1 = kornia.geometry.resize(images1, self.RESIZE, antialias=True)
        images2 = kornia.geometry.resize(images2, self.RESIZE, antialias=True)
        inputs = {
            'image0': kornia.color.rgb_to_grayscale(images1).to(self.device),
            'image1': kornia.color.rgb_to_grayscale(images2).to(self.device),
        }
        with torch.inference_mode():
            outputs_matcher = self.matcher(inputs)
        outputs = {}
        for k, tensor in outputs_matcher.items():
            outputs[k] = tensor[outputs_matcher['confidence'] > confidence_threshold].cpu().numpy()
        
        if ransac:
            _, inliers = fundamental_matrix(outputs['keypoints0'], outputs['keypoints1'])
            for k, v in outputs.items():
                outputs[k] = v[inliers.reshape(-1)]

        if visualize:
            visualized = visualize_matches(
                kornia.tensor_to_image(images1),
                kornia.tensor_to_image(images2),
                outputs['keypoints0'], 
                outputs['keypoints1'], 
                torch.ones(len(outputs['keypoints0']), 1).bool(),
            )
            outputs['visualized'] = visualized

        return outputs


if __name__ == '__main__':
    model = ModelLoFTR(OmegaConf.create({'mode': 'outdoor'}))
    image0 = np.asarray(Image.open('tests/kn_church-2.jpg'))
    image1 = np.asarray(Image.open('tests/kn_church-8.jpg'))
    matches = model(image0, image1)
    for k, v in matches.items():
        if isinstance(v, np.ndarray):
            print(f'{k}: {v.shape}')
    
    visualized = matches['visualized']
    visualized.save('tests/loftr_matches.png')