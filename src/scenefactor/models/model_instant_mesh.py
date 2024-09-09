import subprocess
from pathlib import Path

import numpy as np
import trimesh
from trimesh.base import Trimesh
from PIL import Image
from omegaconf import OmegaConf

from scenefactor.data.common import NumpyTensor
from scenefactor.data.mesh import read_mesh


class ModelInstantMesh:
    """
    """
    TRANSFORM = np.array([
        [ 1,  0,  0,  0],
        [ 0,  0, -1,  0],
        [ 0,  1,  0,  0],
        [ 0,  0,  0,  1],
    ])

    def __init__(self, config: OmegaConf):
        """
        """
        self.config = config

    def __call__(self, image: NumpyTensor['h', 'w', 3]) -> Trimesh:
        """
        """
        script_name   = Path(self.config.script_path).name
        script_parent = Path(self.config.script_path).parent

        Image.fromarray(image).save(script_parent / self.config.image_path)

        subprocess.run(['python3', script_name, 
                        self.config.model_config_path, self.config.image_path], cwd=script_parent)
        subprocess.run(['pkill', '-f', script_name])
        
        return read_mesh(script_parent / self.config.tmesh_path, norm=True, transform=self.TRANSFORM)
    

if __name__ == '__main__':
    model = ModelInstantMesh(OmegaConf.create({
        'script_path': 'third_party/InstantMesh/run.py',
        'model_config_path': 'configs/instant-mesh-large.yaml',
        'image_path': 'examples/input.png',
        'tmesh_path': 'outputs/instant-mesh-large/meshes/input.obj'
    }))

    image = Image.open('third_party/InstantMesh/examples/hatsune_miku.png')
    image = np.asarray(image)
    mesh = model(image)
    mesh.export('tests/instant_mesh.obj')