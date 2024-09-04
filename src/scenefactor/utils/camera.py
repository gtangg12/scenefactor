import numpy as np

from scenefactor.data.common import NumpyTensor
from scenefactor.utils.geom import normalize


def matrix3x4_to_4x4(matrix: NumpyTensor[3, 4]):
    """
    """
    bottom_row = np.tile(np.array([0, 0, 0, 1]), (matrix.shape[0], 1, 1))
    return np.concatenate([matrix, bottom_row], axis=1)


def view_matrix(
    camera_position: NumpyTensor['n...', 3],
    lookat_position: NumpyTensor['n...', 3] = np.array([0, 0, 0]),
    up             : NumpyTensor[3]         = np.array([0, 1, 0])
) -> NumpyTensor['n...', 4, 4]:
    """
    Given lookat position, camera position, and up vector, compute cam2world poses.
    """
    if camera_position.ndim == 1:
        camera_position = camera_position[None, :]
    if lookat_position.ndim == 1:
        lookat_position = lookat_position[None, :]
    camera_position = camera_position.astype(np.float32)
    lookat_position = lookat_position.astype(np.float32)

    cam_u = np.tile(up[None, :], (len(lookat_position), 1)).astype(np.float32)

    # handle degenerate cases
    crossp = np.abs(np.cross(lookat_position - camera_position, cam_u)).max(axis=-1)
    camera_position[crossp < 1e-6] += 1e-6

    cam_z = (lookat_position - camera_position)
    cam_z /= np.linalg.norm(cam_z, axis=-1, keepdims=True)
    cam_x = np.cross(cam_z, cam_u)
    cam_x /= np.linalg.norm(cam_x, axis=-1, keepdims=True)
    cam_y = np.cross(cam_x, cam_z)
    cam_y /= np.linalg.norm(cam_y, axis=-1, keepdims=True)

    poses = np.stack([cam_x, cam_y, -cam_z, camera_position], axis=-1) # nerfstudio convention [right, up, -lookat]
    poses = matrix3x4_to_4x4(poses)
    return poses


if __name__ == '__main__':
    pass