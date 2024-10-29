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


def bmask_sample_points(bmask: NumpyTensor['h', 'w'], num_samples: int, indexing='ij') -> NumpyTensor['n', 2]:
    """
    Sample points from binary mask.
    """
    assert indexing in ['ij', 'xy']
    indices = np.where(bmask)
    indices = np.array(indices).T
    indices = indices[np.random.choice(len(indices), num_samples, replace=False)]
    if indexing == 'xy':
        indices = indices[:, [1, 0]]
    return indices


def bmask_sample_points_grid(bmask: NumpyTensor['h', 'w'], stride: int, std=0) -> NumpyTensor['n', 2]:
    """
    Sample points from binary mask on a grid.
    """
    H, W = bmask.shape
    points_x, points_y = np.meshgrid(np.arange(0, W, stride), np.arange(0, H, stride))
    points = np.stack([points_x.flatten(), points_y.flatten()], axis=1)
    if std:
        points = points + np.random.normal(0, std, points.shape)
        points = np.clip(points, 0, [W - 1, H - 1])
        points = points.astype(int)
    points = points[bmask[
        points[:, 1].astype(int), 
        points[:, 0].astype(int),
    ]]
    return points    


def bmask_iou(bmask1: NumpyTensor['h', 'w'], bmask2: NumpyTensor['h', 'w']) -> float:
    """
    Compute intersection over union of binary masks.
    """
    return np.sum(bmask1 & bmask2) / np.sum(bmask1 | bmask2)


def resize_bmask(bmask: NumpyTensor['h', 'w'], size: tuple[int, int]) -> NumpyTensor['h', 'w']:
    """
    Resize binary mask.
    """
    return cv2.resize(bmask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)


def dialate_bmask(bmask: NumpyTensor['h', 'w'], radius) -> NumpyTensor['h', 'w']:
    """
    """
    bmask = bmask.astype(np.uint8)
    bmask = cv2.dilate(bmask, np.ones((radius, radius)), iterations=1)
    return bmask.astype(bool)


def erode_bmask(bmask: NumpyTensor['h', 'w'], radius) -> NumpyTensor['h', 'w']:
    """
    """
    bmask = bmask.astype(np.uint8)
    bmask = cv2.erode(bmask, np.ones((radius, radius)), iterations=1)
    return bmask.astype(bool)


def bmask_boundary_length(bmask: NumpyTensor['h', 'w']) -> int:
    """
    """
    lmask = dialate_bmask(bmask, radius=2)
    return np.sum(lmask & ~bmask)


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


def connected_components(bmask: NumpyTensor['h', 'w'], return_background=False) -> NumpyTensor['n', 'h', 'w']:
    """
    """
    regions = cv2.connectedComponents(bmask.astype(np.uint8), connectivity=8)[1]
    if not return_background:
        regions = decompose_cmask(regions)[1:] # ignore background
    return regions


def compute_holes(bmask: NumpyTensor['h', 'w']) -> list[tuple[NumpyTensor['h', 'w'], int]]:
    """
    Computes the holes in a binary mask. Assumes (0, 0) is background (can check with compute_bbox).
    """
    bmask = ~bmask
    nregions, regions, stats, _ = cv2.connectedComponentsWithStats(bmask.astype(np.uint8), 8)
    region_bmasks = decompose_cmask(regions)
    holes = []
    for i in range(1, nregions): # index 0 refers to all pixels with label 0 (object)
        if region_bmasks[i][0, 0]: # TODO: don't assume background defined as having same value as (0, 0)
            continue
        holes.append((region_bmasks[i], stats[i, cv2.CC_STAT_AREA])) # (region bmask, area)
    return holes


def remove_background(
    image: NumpyTensor['h', 'w', 3], 
    bmask: NumpyTensor['h', 'w'], 
    background=0, outline_thickness=1
) -> NumpyTensor['h', 'w', 3]:
    """
    """
    image = image.copy()
    image[~bmask] = background
    if outline_thickness:
        contours, _ = cv2.findContours(bmask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (0, 0, 0), outline_thickness)
    return image


def fill_background_holes(cmask: NumpyTensor['h', 'w'], max_area=4096, background=0) -> NumpyTensor['h', 'w']:
    """
    """
    cmask_filled = np.array(cmask)
    for label in np.unique(cmask):
        if label == background:
            continue
        for hole, area in compute_holes(cmask == label):
            if area > max_area:
                continue
            hole_labels, hole_counts = np.unique(cmask[hole], return_counts=True)
            if hole_labels[np.argmax(hole_counts)] == background:
                cmask_filled[hole] = label
    return cmask_filled


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


def bbox_overlap(bbox1: BBox, bbox2: BBox) -> float:
    """
    """
    intersection = bbox_intersection(bbox1, bbox2)
    if intersection is None:
        return 0
    return bbox_area(intersection) / bbox_area(bbox1)


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
    rcenter = (rmin + rmax) / 2
    ccenter = (cmin + cmax) / 2
    rsize = (rmax - rmin) * mult
    csize = (cmax - cmin) * mult
    rmin, rmax = rcenter - rsize / 2, rcenter + rsize / 2
    cmin, cmax = ccenter - csize / 2, ccenter + csize / 2
    return int(rmin), int(cmin), int(rmax), int(cmax)


def deduplicate_bboxes(bboxes: list[BBox], iou=0.85, return_indices=False) -> BBox:
    """
    Given bounding boxes `bboxes` in LRTB format, deduplicate them as defined by IoU threshold.
    """
    bboxes_dedup, indices = [bboxes[0]], [0]
    for i, bbox in enumerate(bboxes[1:]):
        ious = [bbox_iou(bbox, bb) for bb in bboxes_dedup]
        if max(ious) < iou:
            bboxes_dedup.append(bbox), indices.append(i + 1)
    bboxes_dedup, indices = np.stack(bboxes_dedup), np.array(indices)
    if return_indices:
        return bboxes_dedup, indices
    return bboxes_dedup