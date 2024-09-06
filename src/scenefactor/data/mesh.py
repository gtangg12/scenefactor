from pathlib import Path

import numpy as np
import trimesh
from trimesh.base import Trimesh, Scene

from scenefactor.data.common import NumpyTensor
from scenefactor.utils.geom import homogeneous_transform, homogeneous_transform_handle_small, bounding_box_centroid


def scene2mesh(scene: Scene, process=True) -> Trimesh:
    """
    Converts scene with multiple geometries into a single mesh with default coordinate convention.
    """
    if len(scene.geometry) == 0:
        raise ValueError('Scene has no geometry')
    else:
        data = []
        for name, geom in scene.geometry.items():
            if name in scene.graph:
                transform, _ = scene.graph[name]
                transform = homogeneous_transform_handle_small(transform)
                vertices = homogeneous_transform(transform, geom.vertices)
            else:
                vertices = geom.vertices
            # process=True removes duplicate vertices (needed for correct face graph), 
            # affecting face indices but not faces.shape
            data.append(Trimesh(vertices=vertices, faces=geom.faces, visual=geom.visual, process=process))

        mesh = trimesh.util.concatenate(data)
        mesh = Trimesh(vertices=mesh.vertices, faces=mesh.faces, visual=mesh.visual, process=process)
    return mesh


def norm_mesh(mesh: Trimesh) -> Trimesh:
    """
    Returns new mesh with vertices normalized to the bounding box [-1, 1].
    """
    mesh = mesh.copy()
    centroid = bounding_box_centroid(mesh.vertices)
    mesh.vertices -= centroid
    mesh.vertices /= np.abs(mesh.vertices).max()
    mesh.vertices *= (1 - 1e-3)
    return mesh


def read_mesh(filename: Path, norm=False, process=True, transform: NumpyTensor[4, 4]=None) -> Trimesh:
    """
    Read mesh/convert a possible scene to mesh. 
    
    If conversion occurs, the returned mesh has only vertex and face data i.e. no texture information.

    NOTE: process=True may led to unexpected outcomes, such as face color misalignment with faces
    """    
    source = trimesh.load(filename)

    if isinstance(source, trimesh.Scene):
        mesh = scene2mesh(source, process=process)
    else:
        assert(isinstance(source, trimesh.Trimesh))
        mesh = source
    if norm:
        mesh = norm_mesh(mesh)
    if transform is not None:
        mesh.vertices = homogeneous_transform(transform, mesh.vertices)
    return mesh