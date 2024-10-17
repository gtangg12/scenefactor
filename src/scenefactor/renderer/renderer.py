import pyrender
import trimesh

import numpy as np
from omegaconf import OmegaConf
from trimesh.base import Trimesh
from pyrender.shader_program import ShaderProgramCache as DefaultShaderCache
from tqdm import tqdm

from scenefactor.data.common import NumpyTensor
from scenefactor.data.sequence import FrameSequence
from scenefactor.utils.camera import unpack_camera_intrinsics_fov
from scenefactor.utils.visualize import visualize_depth


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
        def default_camera_params():
            """ 
            Returns camera parameters for a 90 degree FOV camera.
            """
            params = unpack_camera_intrinsics_fov(np.deg2rad(90), *self.config.target_dim)
            params['width']  = self.config.target_dim[0]
            params['height'] = self.config.target_dim[1]
            return params
        
        self.camera_params = camera_params or default_camera_params()
        self.camera = pyrender.IntrinsicsCamera(**{
            'fx': self.camera_params['fx'],
            'fy': self.camera_params['fy'],
            'cx': self.camera_params['cx'],
            'cy': self.camera_params['cy'],
        })
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
) -> list[dict]:
    """
    """
    renderer.set_object(source)
    renderer.set_camera(camera_params)
    outputs = []
    for pose in tqdm(poses, desc='Rendering views...'):
        renders = {**renderer.render(pose, shaders), 'pose': pose} 
        outputs.append(renders)
    outputs = {
        k: np.stack([x[k] for x in outputs], axis=0) for k in outputs[0].keys()
    }
    outputs['camera_params'] = renderer.camera_params
    return outputs


def renders_to_sequence(renders: list[dict]):
    """
    """
    return FrameSequence(
        images=renders['image'],
        depths=renders['depth'],
        poses =renders['pose'],
        metadata={'camera_params': renders['camera_params']}
    )


if __name__ == '__main__':
    import os
    from PIL import Image
    from scenefactor.utils.mesh import read_mesh
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
