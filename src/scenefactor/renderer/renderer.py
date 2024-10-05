import pyrender
import trimesh

import numpy as np
from omegaconf import OmegaConf
from trimesh.base import Trimesh
from pyrender.shader_program import ShaderProgramCache as DefaultShaderCache
from tqdm import tqdm

from scenefactor.data.common import NumpyTensor
from scenefactor.utils.visualize import visualize_depth


DEFAULT_CAMERA_PARAMS = {'fov': 60, 'znear': 0.01, 'zfar': 16}


RenderObject = Trimesh | pyrender.Scene


class Renderer:
    """
    """
    def __init__(self, config: OmegaConf):
        """
        """
        self.config = config
        self.renderer = pyrender.OffscreenRenderer(*config.target_dim)
        self.shaders = {
            'default': (DefaultShaderCache(), self.postprocess_shader_default),
        }

    def set_object(self, source: RenderObject, ambient_light=[1.0, 1.0, 1.0]):
        """
        """
        if isinstance(source, pyrender.Scene):
            self.scene = source
        elif isinstance(source, Trimesh):
            scene = pyrender.Scene(ambient_light=ambient_light)
            scene.add(pyrender.Mesh.from_trimesh(source))
            self.scene = scene
        else:
            raise ValueError('Invalid source type. Must be either Trimesh or pyrender.Scene')

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
        def rasterize(shader_name: str):
            """
            """
            self.renderer._renderer._program_cache, postprocess_func = self.shaders[shader_name]
            output = self.renderer.render(self.scene)
            output = postprocess_func(*output)
            return output
        
        self.scene.set_pose(self.camera_node, pose)
        outputs = {}
        for shader_name in shaders:
            outputs.update(**rasterize(shader_name))
        return outputs
    
    def postprocess_shader_default(
        self,
        image: NumpyTensor['h', 'w', 3],
        depth: NumpyTensor['h', 'w']
    ) -> dict:
        """
        """
        return {'image': image, 'depth': depth}


def render_multiview(
    source: RenderObject, renderer: Renderer, poses: NumpyTensor['n', '4', '4'], shaders=['default'], camera_params=None
) -> list[NumpyTensor['h', 'w', 3]]:
    """
    """
    renderer.set_object(source)
    renderer.set_camera(camera_params)
    outputs = []
    for pose in tqdm(poses, desc='Rendering views...'):
        outputs.append(renderer.render(pose, shaders))
    return outputs


if __name__ == '__main__':
    import os
    from PIL import Image
    from scenefactor.data.mesh import read_mesh
    from scenefactor.utils.camera_generation import sample_view_matrices_method, sample_view_matrices_spherical

    mesh = read_mesh('tests/instant_mesh.obj', norm=True, transform=np.array([
        [ 1,  0,  0,  0],
        [ 0,  0, -1,  0],
        [ 0,  1,  0,  0],
        [ 0,  0,  0,  1],
    ]))
    poses, camera_params = sample_view_matrices_method('icosahedron', radius=2), None
    renderer = Renderer(OmegaConf.create({'target_dim': (640, 480)}))
    outputs = render_multiview(mesh, renderer, poses, camera_params=camera_params)
    
    path = 'tests/renderer'
    os.makedirs(path, exist_ok=True)
    for i, renders in enumerate(outputs):
        image = Image.fromarray(renders['image'])
        depth = visualize_depth(renders['depth'])
        image.save(f'{path}/image_{i}.png')
        depth.save(f'{path}/depth_{i}.png')
