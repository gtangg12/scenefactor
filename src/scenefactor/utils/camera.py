import numpy as np

from scenefactor.data.common import NumpyTensor


def normalize(x, p=2, dim=0, eps=1e-12):
    """
    Equivalent to torch.nn.functional.normalize.
    """
    norm = np.linalg.norm(x, ord=p, axis=dim, keepdims=True)
    return x / (norm + eps)


def matrix3x4_to_4x4(matrix3x4: NumpyTensor[4, 4]) -> NumpyTensor[4, 4]:
    """
    Convert a 3x4 transformation matrix to a 4x4 transformation matrix.
    """
    bottom = np.zeros_like(matrix3x4[:, 0, :].unsqueeze(-2))
    bottom[..., -1] = 1
    return np.concatenate([matrix3x4, bottom], dim=-2)


def view_matrix(
    camera_position: NumpyTensor['n... 3'],
    lookat_position: NumpyTensor['n... 3'] = np.array([0, 0, 0]),
    up             : NumpyTensor['3']      = np.array([0, 1, 0]),
) -> NumpyTensor[4, 4]:
    """
    Given lookat position, camera position, and up vector, compute cam2world poses.
    """
    if camera_position.ndim == 1:
        camera_position = camera_position.unsqueeze(0)
    if lookat_position.ndim == 1:
        lookat_position = lookat_position.unsqueeze(0)
    camera_position = camera_position.float()
    lookat_position = lookat_position.float()

    cam_u = up.unsqueeze(0).repeat(len(lookat_position), 1).float().to(camera_position.device)

    # handle degenerate cases
    crossp = np.abs(np.cross(lookat_position - camera_position, cam_u, dim=-1)).max(dim=-1).values
    camera_position[crossp < 1e-6] += 1e-6

    cam_z = normalize((lookat_position - camera_position), dim=-1)
    cam_x = normalize(np.cross(cam_z, cam_u, dim=-1), dim=-1)
    cam_y = normalize(np.cross(cam_x, cam_z, dim=-1), dim=-1)
    poses = np.stack([cam_x, cam_y, -cam_z, camera_position], dim=-1) # nerfstudio convention [right, up, -lookat]
    poses = matrix3x4_to_4x4(poses)
    return poses


if __name__ == '__main__':
    pass