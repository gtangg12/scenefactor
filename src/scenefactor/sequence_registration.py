import numpy as np
import torch
from trimesh.base import Trimesh
from omegaconf import OmegaConf

from scenefactor.data.common import NumpyTensor
from scenefactor.data.sequence import FrameSequence
from scenefactor.renderer.renderer import Renderer
from scenefactor.models import ModelLoFTR
from scenefactor.utils.camera_generation import sample_view_matrices_method


def compute_homogeneous_transform_plus_scale(
    points0: NumpyTensor['n', 3], 
    points1: NumpyTensor['n', 3]
) -> tuple[NumpyTensor[4, 4], float]:
    """
    Relevant links:
        - http://stackoverflow.com/a/32244818/263061 (solution with scale)
        - "Least-Squares Rigid Motion Using SVD" (no scale but easy proofs and explains how weights could be added)

    Rigidly (+scale) aligns two point clouds with know point-to-point correspondences with least-squares error.
    Returns (scale factor c, rotation matrix R, translation vector t) such that

        Q = P*cR + t
    
    if they align perfectly, or such that

        SUM over point i ( | P_i*cR + t - Q_i |^2 )
    
    is minimised if they don't align perfectly.
    """
    assert points0.shape == points1.shape
    n, _ = points0.shape
    centered_points0 = points0 - points0.mean(axis=0)
    centered_points1 = points1 - points1.mean(axis=0)
    variance_points0 = np.var(points0, axis=0)

    mat = np.dot(centered_points0.T, centered_points1) / n
    V, S, W = np.linalg.svd(mat)
    if (
        np.linalg.det(V) * 
        np.linalg.det(W)
    ) < 0:
        S[-1], V[:, -1] = -S[-1], -V[:, -1]

    # rotation, translation, scale
    R = np.dot(V, W)
    t = points1.mean(axis=0) - points0.mean(axis=0).dot(c * R)
    c = 1 / variance_points0.sum() * np.sum(S)
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3,  3] = t
    return pose, c


class SequenceRegistration:
    """
    """
    def __init__(self, config: OmegaConf):
        """
        """
        self.config = config
        self.renderer = Renderer(config.renderer)
        self.matcher = ModelLoFTR(config.model)

    def __call__(self, mesh: Trimesh, label: int, sequence: FrameSequence) -> tuple[NumpyTensor[4, 4], float]:
        """
        Computes 6DoF transform and scale for mesh registered to object denoted by label in sequence.
        """
        poses = sample_view_matrices_method(
            self.config.camera_generation_method,
            self.config.camera_generation_radius,
        )
        renders = [self.renderer.render(pose) for pose in poses]
        images = np.stack([render['image'] for render in renders])
        depths = np.stack([render['depth'] for render in renders])

        def filter_keypoints(keypoints, bmask):
            return keypoints[bmask[keypoints[:, 1], keypoints[:, 0]]] # only consider keypoints in bmask
        
        max_keypoints, max_matches, max_index = 0, {}, None
        for i, image in enumerate(images):
            for j, frame in enumerate(sequence):
                bmask = frame.imask == label
                matches = self.matcher.process(frame.image, image)
                keypoints0 = filter_keypoints(matches['keypoints0'], bmask)
                keypoints1 = filter_keypoints(matches['keypoints1'], bmask)
                if max_matches is None or len(keypoints0) > max_keypoints:
                    max_keypoints = len(keypoints0)
                    max_matches = {
                        'keypoints0': keypoints0,
                        'keypoints1': keypoints1,
                    }
                    max_index = (i, j)
        
        def compute_directions(camera_params: dict):
            """
            """
            w, h = camera_params['width'], camera_params['height']
            i, j = np.meshgrid(np.arange(w), np.arange(h))
            i = i.reshape(-1)
            j = j.reshape(-1)
            rays = np.stack([
                (i - w / 2) / camera_params['fx'],
                (j - h / 2) / camera_params['fy'],
                np.ones_like(i)
            ], axis=0)
            return rays
        
        def compute_positions(
            keypoints: NumpyTensor['n', 2], depth: NumpyTensor['h', 'w'], pose: NumpyTensor[4, 4], camera_params: dict
        ) -> NumpyTensor['n', 3]:
            """
            """
            positions = depth * compute_directions(camera_params)  + pose[:, 3]
            return positions[keypoints[:, 1], keypoints[:, 0]]

        
        i, j = max_index
        keypoints0 = max_matches['keypoints0']
        keypoints1 = max_matches['keypoints1']
        positions_rendered = compute_positions(keypoints0, depths[i], poses[i], self.config.renderer.camera_params)
        positions_sequence = compute_positions(keypoints1, sequence[j].depth, sequence[j].pose, sequence[j].metadata['camera_params'])
        
        transform, scale = compute_homogeneous_transform_plus_scale(positions_rendered, positions_sequence)
        return transform, scale
    

if __name__ == '__main__':
    pass