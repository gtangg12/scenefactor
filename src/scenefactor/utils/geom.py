import cv2
import numpy as np

from scenefactor.data.common import NumpyTensor


BBox = tuple[int, int, int, int] # TLBR format


def normalize(x, p=2, dim=0, eps=1e-12):
    """
    Equivalent to torch.nn.functional.normalize.
    """
    norm = np.linalg.norm(x, ord=p, axis=dim, keepdims=True)
    return x / (norm + eps)


def homogeneous_transform(transform: NumpyTensor[4, 4], coords: NumpyTensor['n', 3]) -> NumpyTensor['n', 3]:
    """
    Apply homogeneous transformation to coordinates.
    """
    homogeneous = np.concatenate([coords, np.ones((coords.shape[0], 1))], axis=1)
    return (transform @ homogeneous.T).T[:, :3]


def homogeneous_transform_handle_small(transform: NumpyTensor[4, 4]) -> NumpyTensor[4, 4]:
    """
    Handles common case that results from numerical instability.
    """
    identity = np.eye(4)
    if np.allclose(transform, identity, atol=1e-6):
        return identity
    return transform


def bounding_box(coords: NumpyTensor['n', 3]) -> NumpyTensor[2, 3]:
    """
    Compute bounding box from coordinates.
    """
    return np.array([coords.min(axis=0), coords.max(axis=0)])


def bounding_box_centroid(coords: NumpyTensor['n', 3]) -> NumpyTensor[3]:
    """
    Compute bounding box centroid from coordinates.
    """
    return bounding_box(coords).mean(axis=0)


def bmask_sample_points(bmask: NumpyTensor['h', 'w'], num_samples: int) -> NumpyTensor['n', 2]:
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


def combine_bmasks(bmasks: NumpyTensor['n', 'h', 'w'], sort=False) -> NumpyTensor['h w']:
    """
    """
    cmask = np.zeros_like(bmasks[0], dtype=int)
    if sort:
        bmasks = sorted(bmasks, key=lambda x: x.sum(), reverse=True)
    for i, bmask in enumerate(bmasks):
        cmask[bmask] = i + 1
    return cmask


def decompose_cmask(cmask: NumpyTensor['h', 'w'], background=None) -> NumpyTensor['n', 'h', 'w']:
    """
    """
    labels = np.unique(cmask)
    np.sort(labels)
    if background is not None:
        labels = labels[labels != background]
    return cmask == labels[:, None, None]


def deduplicate_bmasks(bmasks: NumpyTensor['n', 'h', 'w'], iou=0.85, return_indices=False) -> NumpyTensor['n', 'h', 'w']:
    """
    Given binary masks `bmasks`, deduplicate them as defined by IoU threshold.
    """
    bmasks_dedup, indices = [bmasks[0]], [0]
    for i, bmask in enumerate(bmasks[1:]):
        ious = [bmask_iou(bmask, bm) for bm in bmasks_dedup]
        if max(ious) < iou:
            bmasks_dedup.append(bmask), indices.append(i + 1)
    bmasks_dedup, indices = np.stack(bmasks_dedup), np.array(indices)
    if return_indices:
        return bmasks_dedup, indices
    return bmasks_dedup


def remove_artifacts(bmask: NumpyTensor['h', 'w'], mode: str, min_area=128) -> NumpyTensor['h', 'w']:
    """
    Removes small islands/fill holes from a mask.
    """
    assert mode in ['holes', 'islands']
    mode_holes = (mode == 'holes')
    bmask = (mode_holes ^ bmask).astype(np.uint8)
    nregions, regions, stats, _ = cv2.connectedComponentsWithStats(bmask, 8)
    sizes = stats[:, -1][1:]  # Row 0 corresponds to 0 pixels
    fill = [i + 1 for i, s in enumerate(sizes) if s < min_area] + [0]
    if not mode_holes:
        fill = [i for i in range(nregions) if i not in fill]
    return np.isin(regions, fill)


def remove_artifacts_cmask(mask: NumpyTensor['h', 'w'], mode: str, min_area=128) -> NumpyTensor['h', 'w']:
    """
    Removes small islands/fill holes from a mask.
    """
    mask_combined = np.zeros_like(mask)
    for label in np.unique(mask):
        mask_combined[remove_artifacts(mask == label, mode=mode, min_area=min_area)] = label
    return mask_combined


def crop(image: NumpyTensor['h', 'w', 'c...'], bbox: BBox) -> NumpyTensor['h', 'w', 'c']:
    """
    """
    return image[bbox[0]:bbox[2], bbox[1]:bbox[3]]


def bbox_check_bounds(bbox: BBox, h: int, w: int) -> bool:
    """
    """
    return bbox[0] >= 0 and bbox[2] <= h and \
           bbox[1] >= 0 and bbox[3] <= w


def bbox_area(bbox: BBox) -> int:
    """
    """
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def bbox_intersection(bbox1: BBox, bbox2: BBox) -> BBox | None:
    """
    """
    rmin = max(bbox1[0], bbox2[0])
    cmin = max(bbox1[1], bbox2[1])
    rmax = min(bbox1[2], bbox2[2])
    cmax = min(bbox1[3], bbox2[3])
    if rmin >= rmax or cmin >= cmax:
        return None
    return rmin, cmin, rmax, cmax


def bbox_union(bbox1: BBox, bbox2: BBox) -> BBox:
    """
    """
    return (
        min(bbox1[0], bbox2[0]),
        min(bbox1[1], bbox2[1]),
        max(bbox1[2], bbox2[2]),
        max(bbox1[3], bbox2[3])
    )


def bbox_iou(bbox1: BBox, bbox2: BBox) -> float:
    """
    """
    intersection = bbox_intersection(bbox1, bbox2)
    if intersection is None:
        return 0
    union = bbox_union(bbox1, bbox2)
    return bbox_area(intersection) / bbox_area(union)


def compute_bbox(bmask: NumpyTensor['h', 'w']) -> BBox:
    """
    Returns the bounding box of a binary mask, where the minimum is inclusive and the maximum is exclusive.
    """
    rows = np.any(bmask, axis=1)
    cols = np.any(bmask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, cmin, rmax + 1, cmax + 1


def resize_bbox(bbox: BBox, mult: float) -> BBox:
    """
    Expands a bounding box by a factor of expand_mult.

    NOTE: does not check if the expanded bbox is within the image bounds.
    """
    rmin, cmin, rmax, cmax = bbox
    rcenter = (rmin + rmax) // 2
    ccenter = (cmin + cmax) // 2
    rsize = int((rmax - rmin) * mult)
    csize = int((cmax - cmin) * mult)
    rmin, rmax = rcenter - rsize // 2, rcenter + rsize // 2
    cmin, cmax = ccenter - csize // 2, ccenter + csize // 2
    return rmin, cmin, rmax, cmax


def deduplicate_bboxes(bboxes: list[BBox], iou=0.85, return_indices=False) -> BBox:
    """
    Given bounding boxes `bboxes` in LRTB format, deduplicate them as defined by IoU threshold.
    """
    bboxes_dedup, indices = [bboxes[0]], [0]
    for i, bbox in enumerate(bboxes[1:]):
        print(bboxes_dedup)
        ious = [bbox_iou(bbox, bb) for bb in bboxes_dedup]
        if max(ious) < iou:
            bboxes_dedup.append(bbox), indices.append(i + 1)
    bboxes_dedup, indices = np.stack(bboxes_dedup), np.array(indices)
    if return_indices:
        return bboxes_dedup, indices
    return bboxes_dedup