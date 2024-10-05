from pathlib import Path

import numpy as np
import trimesh
from trimesh.base import Trimesh, Scene

from scenefactor.data.common import NumpyTensor
from scenefactor.utils.geom import homogeneous_transform, homogeneous_transform_handle_small, bounding_box_centroid


def scene2tmesh(scene: Scene, process=True) -> Trimesh:
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

        tmesh = trimesh.util.concatenate(data)
        tmesh = Trimesh(vertices=tmesh.vertices, faces=tmesh.faces, visual=tmesh.visual, process=process)
    return tmesh


def norm_tmesh(tmesh: Trimesh) -> Trimesh:
    """
    Returns new tmesh with vertices normalized to the bounding box [-1, 1].
    """
    tmesh = tmesh.copy()
    centroid = bounding_box_centroid(tmesh.vertices)
    tmesh.vertices -= centroid
    tmesh.vertices /= np.abs(tmesh.vertices).max()
    tmesh.vertices *= (1 - 1e-3)
    return tmesh


def read_tmesh(filename: Path, norm=False, process=True, transform: NumpyTensor[4, 4]=None) -> Trimesh:
    """
    Read tmesh/convert a possible scene to tmesh. 
    
    If conversion occurs, the returned mesh has only vertex and face data i.e. no texture information.

    NOTE: process=True may led to unexpected outcomes, such as face color misalignment with faces
    """    
    source = trimesh.load(filename)

    if isinstance(source, trimesh.Scene):
        tmesh = scene2tmesh(source, process=process)
    else:
        assert(isinstance(source, trimesh.Trimesh))
        tmesh = source
    if norm:
        tmesh = norm_tmesh(tmesh)
    if transform is not None:
        tmesh.vertices = homogeneous_transform(transform, tmesh.vertices)
    return tmesh