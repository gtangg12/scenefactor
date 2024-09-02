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


def bmask_sample_points(bmask: NumpyTensor['h', 'w'], num_samples: int) -> NumpyTensor['n 2']:
    """
    Sample points from binary mask.
    """
    indices = np.where(bmask)
    indices = np.array(indices).T
    indices = indices[np.random.choice(len(indices), num_samples, replace=False)]
    return indices


def bmask_iou(bmask1: NumpyTensor['h', 'w'], bmask2: NumpyTensor['h', 'w']) -> float:
    """
    Compute intersection over union of binary masks.
    """
    return np.sum(bmask1 & bmask2) / np.sum(bmask1 | bmask2)


def combine_bmasks(masks: NumpyTensor['n h w'], sort=False) -> NumpyTensor['h w']:
    """
    """
    mask_combined = np.zeros_like(masks[0], dtype=int)
    if sort:
        masks = sorted(masks, key=lambda x: x.sum(), reverse=True)
    for i, mask in enumerate(masks):
        mask_combined[mask] = i + 1
    return mask_combined


def decompose_mask(mask: NumpyTensor['h w'], background=0) -> NumpyTensor['n h w']:
    """
    """
    labels = np.unique(mask)
    labels = labels[labels != background]
    return mask == labels[:, None, None]