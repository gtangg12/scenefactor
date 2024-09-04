import numpy as np

from scenefactor.data.common import NumpyTensor
from scenefactor.utils.camera import view_matrix


def golden_ratio():
    return (1 + np.sqrt(5)) / 2


def tetrahedron():
    return np.array([
        [ 1,  1,  1],
        [-1, -1,  1],
        [-1,  1, -1],
        [ 1, -1, -1],
    ])


def octohedron():
    return np.array([
        [ 1,  0,  0],
        [ 0,  0,  1],
        [-1,  0,  0],
        [ 0,  0, -1],
        [ 0,  1,  0],
        [ 0, -1,  0],
    ])


def cube():
    return np.array([
        [ 1,  1,  1],
        [-1,  1,  1],
        [-1, -1,  1],
        [ 1, -1,  1],
        [ 1,  1, -1],
        [-1,  1, -1],
        [-1, -1, -1],
        [ 1, -1, -1],
    ])


def icosahedron():
    phi = golden_ratio()
    return np.array([
        [-1,  phi,  0],
        [-1, -phi,  0],
        [ 1,  phi,  0],
        [ 1, -phi,  0],
        [ 0, -1,  phi],
        [ 0,  1,  phi],
        [ 0, -1, -phi],
        [ 0,  1, -phi],
        [ phi,  0, -1],
        [ phi,  0,  1],
        [-phi,  0, -1],
        [-phi,  0,  1],
    ]) / np.sqrt(1 + phi ** 2)


def dodecahedron():
    phi = golden_ratio()
    a, b = 1 / phi, 1 / (phi * phi)
    return np.array([
        [-a, -a,  b], [ a, -a,  b], [ a,  a,  b], [-a,  a,  b],
        [-a, -a, -b], [ a, -a, -b], [ a,  a, -b], [-a,  a, -b],
        [ b, -a, -a], [ b,  a, -a], [ b,  a,  a], [ b, -a,  a],
        [-b, -a, -a], [-b,  a, -a], [-b,  a,  a], [-b, -a,  a],
        [-a,  b, -a], [ a,  b, -a], [ a,  b,  a], [-a,  b,  a],
    ]) / np.sqrt(a ** 2 + b ** 2)


def standard(n=8, elevation=15):
    """
    """
    pphi =  elevation * np.pi / 180
    nphi = -elevation * np.pi / 180
    coords = []
    for phi in [pphi, nphi]:
        for theta in np.linspace(0, 2 * np.pi, n, endpoint=False):
            coords.append([
                np.cos(theta) * np.cos(phi),
                np.sin(phi),
                np.sin(theta) * np.cos(phi),
            ])
    coords.append([0,  0,  1])
    coords.append([0,  0, -1])
    return np.array(coords)


def swirl(n=120, cycles=1, elevation_range=(-45, 60)):
    """
    """
    pphi = elevation_range[0] * np.pi / 180
    nphi = elevation_range[1] * np.pi / 180
    thetas = np.linspace(0, 2 * np.pi, n, endpoint=False)
    coords = []
    for i, phi in enumerate(np.linspace(pphi, nphi, n)):
        coords.append([
            np.cos(cycles * thetas[i]) * np.cos(phi),
            np.sin(phi),
            np.sin(cycles * thetas[i]) * np.cos(phi),
        ])
    return np.array(coords)


def sample_view_matrices(
    n: int, radius: float, lookat_position: NumpyTensor['3']=np.array([0, 0, 0])
) -> NumpyTensor['n', 4, 4]:
    """
    Sample n uniformly distributed view matrices spherically with given radius.
    """
    tht = np.random.rand(n) * np.pi * 2
    phi = np.random.rand(n) * np.pi
    world_x = radius * np.sin(phi) * np.cos(tht)
    world_y = radius * np.sin(phi) * np.sin(tht)
    world_z = radius * np.cos(phi)
    camera_position = np.stack([world_x, world_y, world_z], dim=-1)
    lookat_position = lookat_position.unsqueeze(0).repeat(n, 1)
    return view_matrix(
        camera_position, lookat_position, up=np.array([0, 1, 0])
    )


def sample_view_matrices_method(
    method: str, radius: float, lookat_position: NumpyTensor['3']=np.array([0, 0, 0]), **kwargs
) -> NumpyTensor['n', 4, 4]:
    """
    Sample view matrices according to a method with given radius.
    """
    camera_position = eval(method)(**kwargs) * radius
    return view_matrix(
        camera_position, lookat_position, up=np.array([0, 1, 0])
    )


if __name__ == '__main__':
    pass