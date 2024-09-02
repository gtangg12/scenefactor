import numpy as np

from scenefactor.data.common import NumpyTensor


def normalize(x, p=2, dim=0, eps=1e-12):
    """
    Equivalent to torch.nn.functional.normalize.
    """
    norm = np.linalg.norm(x, ord=p, axis=dim, keepdims=True)
    return x / (norm + eps)


def homogeneous_transform(transform: NumpyTensor['4 4'], coords: NumpyTensor['n 3']) -> NumpyTensor['n 3']:
    """
    Apply homogeneous transformation to coordinates.
    """
    homogeneous = np.concatenate([coords, np.ones((coords.shape[0], 1))], axis=1)
    return (transform @ homogeneous.T).T[:, :3]


def homogeneous_transform_handle_small(transform: NumpyTensor['4 4']) -> NumpyTensor['4 4']:
    """
    Handles common case that results from numerical instability.
    """
    identity = np.eye(4)
    if np.allclose(transform, identity, atol=1e-6):
        return identity
    return transform


def bounding_box(coords: NumpyTensor['n 3']) -> NumpyTensor['2 3']:
    """
    Compute bounding box from coordinates.
    """
    return np.array([coords.min(axis=0), coords.max(axis=0)])


def bounding_box_centroid(coords: NumpyTensor['n 3']) -> NumpyTensor['3']:
    """
    Compute bounding box centroid from coordinates.
    """
    return bounding_box(coords).mean(axis=0)