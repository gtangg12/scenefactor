import torch
from nerfstudio.cameras.cameras import CAMERA_MODEL_TO_TYPE, Cameras, camera_utils
from nerfstudio.data.scene_box import SceneBox

from scenefactor.data.common import NumpyTensor, TorchTensor
from scenefactor.data.sequence import FrameSequence


def make_scene_box(scale=1):
    """
    """
    return SceneBox(aabb=torch.tensor([
        [-scale, -scale, -scale], 
        [ scale,  scale,  scale]
    ]))


def make_cameras_from_sequence(poses: NumpyTensor[..., 4, 4], sequence: FrameSequence) -> Cameras:
    """
    """ 
    CAMERA_INTRINSICS = ['fx', 'fy', 'cx', 'cy', 'height', 'width']
    CAMERA_INTRINSICS_DISTORTION = ['k1', 'k2', 'k3', 'k4', 'p1', 'p2']
    if 'frame_metadata' in sequence.metadata:
        frame_metadata = [sequence.metadata.frame_metadata[i] for i in range(len(sequence))]
    else:
        frame_metadata = [{} for _ in range(len(sequence))]

    camera_params = {
        'camera_type': CAMERA_MODEL_TO_TYPE[sequence.metadata['camera_params'].get('camera_model', 'PINHOLE')],
        'camera_to_worlds': poses[:, :3, :4],
    }
    for x in CAMERA_INTRINSICS:
        if x in sequence.metadata['camera_params']:
            camera_params[x] = sequence.metadata['camera_params'][x]
        else:
            camera_params[x] = torch.tensor([metadata[x] for metadata in frame_metadata])
    camera_params.update({
        'height': int(camera_params.pop('height')),
        'width' : int(camera_params.pop('width' )),
    })
    
    load_distortion = lambda params: camera_utils.get_distortion_params(
        **{x: params.get(x, 0.0) for x in CAMERA_INTRINSICS_DISTORTION}
    )
    distortion_fixed = any([x in sequence.metadata['camera_params'] for x in CAMERA_INTRINSICS_DISTORTION])
    if distortion_fixed:
        camera_params['distortion_params'] = load_distortion(sequence.metadata['camera_params'])
    else:
        camera_params['distortion_params'] = torch.stack([load_distortion(metadata) for metadata in frame_metadata])
    
    return Cameras(**camera_params)


def transform_and_scale(
    poses: TorchTensor[..., 4, 4], scale: float, transform: TorchTensor[4, 4]
) -> TorchTensor[..., 4, 4]:
    """
    """
    poses = transform @ poses
    poses[:, :3, 3] *= scale
    return poses