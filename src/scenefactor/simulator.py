import numpy as np
from omegaconf import OmegaConf
from trimesh.base import Trimesh

from scenefactor.data.common import NumpyTensor


class Simulator:
    """
    """
    def __init__(self, config: OmegaConf):
        """
        """
        self.config = config

    def add_object(self, name: str, tmesh: Trimesh, pose: NumpyTensor[4, 4]):
        """
        """
        pass

    def get_object(self, name: str) -> dict:
        """
        """
        pass

    def transform_object(self, name: str, transform: NumpyTensor[4, 4]=np.eye(4), scale=1):
        """
        """
        pass


if __name__ == '__main__':
    from scenefactor.renderer.renderer import Renderer

    scene = Simulator(OmegaConf.create({}))
    renderer = Renderer(OmegaConf.create({}))
    renderer.set_scene(scene)

    # update scene wth transforms
    # call renderer.render(pose) to get image