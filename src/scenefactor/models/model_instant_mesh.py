import subprocess
from pathlib import Path

import numpy as np
import trimesh
from trimesh.base import Trimesh
from PIL import Image
from omegaconf import OmegaConf

from scenefactor.data.common import NumpyTensor
from scenefactor.utils.mesh import read_tmesh
from scenefactor.utils.tensor import read_image, untile

PATH = Path(__file__).parents[3] / 'third_party/InstantMesh'

SCRIPT = PATH / 'run.py'

VIEW_SIZE = 320
VIEW_ROWS = 960 // VIEW_SIZE
VIEW_COLS = 640 // VIEW_SIZE


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
        self.input_name = Path(self.config.input_path).name
        self.model_name = Path(self.config.model_config).stem

    def __call__(
        self, image: NumpyTensor['h', 'w', 3], return_multiview=False
    ) -> Trimesh | tuple[Trimesh, NumpyTensor['n', 'h', 'w', 3]]:
        """
        """
        Image.fromarray(image).save(PATH / self.config.input_path)
        
        subprocess.run(['python3', SCRIPT, self.config.model_config, self.config.input_path], cwd=PATH)
        subprocess.run(['pkill', '-f', SCRIPT])
        
        tmesh = read_tmesh(PATH / self.config.tmesh_path, norm=True, transform=self.TRANSFORM)
        mview = read_image(PATH / 'outputs' / self.model_name / 'images' / self.input_name)
        mview = untile(mview, VIEW_ROWS, VIEW_COLS)
        if return_multiview:
            return tmesh, mview
        return tmesh


if __name__ == '__main__':
    model = ModelInstantMesh(OmegaConf.create({
        'model_config': 'configs/instant-mesh-large.yaml',
        'input_path': 'examples/input.png',
        'tmesh_path': 'outputs/instant-mesh-large/meshes/input.obj'
    }))

    image = Image.open('third_party/InstantMesh/examples/hatsune_miku.png')
    image = np.asarray(image)
    mesh, mview = model(image, return_multiview=True)
    mesh.export('tests/instant_mesh.obj')
    for i, view in enumerate(mview):
        Image.fromarray(view).save(f'tests/instant_mesh_view_{i}.png')