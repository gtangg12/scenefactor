import pyrender
import trimesh

import numpy as np
from omegaconf import OmegaConf
from trimesh.base import Trimesh
from pyrender.shader_program import ShaderProgramCache as DefaultShaderCache

from scenefactor.data.common import NumpyTensor


DEFAULT_CAMERA_PARAMS = {'fov': 60, 'znear': 0.01, 'zfar': 16}


class Renderer:
    """
    """
    def __init__(self, config: OmegaConf):
        """
        """
        self.config = config
        self.renderer = pyrender.OffscreenRenderer(*config.target_dim)
        self.shaders = {
            'default': (DefaultShaderCache(), None),
        }

    def set_object(self, tmesh: Trimesh):
        """
        """
        self.tmesh = tmesh
        self.scene = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0])
        self.scene.add(pyrender.Mesh.from_trimesh(tmesh))

    def set_camera(self, camera_params: dict = None):
        """
        """
        self.camera_params = camera_params or dict(DEFAULT_CAMERA_PARAMS)
        self.camera_params['yfov'] = self.camera_params.get('yfov', self.camera_params.pop('fov'))
        self.camera_params['yfov'] = self.camera_params['yfov'] * np.pi / 180.0
        self.camera = pyrender.PerspectiveCamera(**self.camera_params)
        self.camera_node = self.scene.add(self.camera)

    def render(self, pose: NumpyTensor['4 4'], shaders=['default']) -> dict:
        """
        """
        def rasterize(shader: str, postprocess: callable = None):
            """
            """
            self.renderer._renderer._program_cache = self.shaders[shader]
            output = self.renderer.render(self.scene)
            if postprocess is not None:
                output = postprocess(output)
            return output
        
        self.scene.set_pose(self.camera_node, pose)
        outputs = {}
        for name, (shader, postprocess) in shaders.items():
            outputs[name] = rasterize(shader, postprocess)
        return outputs
    

if __name__ == '__main__':
    pass