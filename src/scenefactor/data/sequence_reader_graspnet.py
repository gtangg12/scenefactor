import json
from pathlib import Path
from glob import glob
from natsort import natsorted

from scenefactor.data.sequence import FrameSequence
from scenefactor.data.sequence_reader_base import *
from scenefactor.utils.camera import unpack_camera_intrinsics


class GraspNetFrameSequenceReader(FrameSequenceReader):
    """
    """
    READER_CONFIG = 'dataconfig_graspnet.yaml'

    def __init__(self, base_dir: Path | str, name: str, camera='realsense'):
        """
        """
        assert camera in ['realsense', 'kinect']
        super().__init__(base_dir, name)
        self.data_dir = self.base_dir / 'scenes' / name / camera

    def read(self, slice=(0, -1, 20), resize: tuple[int, int]=None) -> FrameSequence:
        """
        """
        def read_filenames(pattern: str) -> list[str]:
            """
            """
            filenames = natsorted(glob(f'{self.data_dir}/{pattern}'))
            filenames = filenames[slice[0]:slice[1]:slice[2]]
            return filenames

        def read_poses() -> NumpyTensor['n', 4, 4]:
            """
            """
            poses = np.load(self.data_dir / 'camera_poses.npy')
            poses = poses[slice[0]:slice[1]:slice[2]]
            poses = poses @ np.array(self.metadata['pose_axis_transform'])
            return poses

        def read_camera_params() -> dict:
            """
            """
            return unpack_camera_intrinsics(np.load(self.data_dir / 'camK.npy'))

        image_filenames = read_filenames('rgb/*.png')
        depth_filenames = read_filenames('depth/*.png')
        imask_filenames = read_filenames('label/*.png')

        poses = read_poses()

        sequence = FrameSequence(
            images=np.array([self.load_image(f, resize) for f in image_filenames]),
            depths=np.array([self.load_depth(f, resize, scale=self.metadata['depth_scale']) for f in depth_filenames]),
            imasks=np.array([self.load_imask(f, resize) for f in imask_filenames]),
            poses=poses,
            metadata=self.metadata
        )
        H = sequence.images[0].shape[0]
        W = sequence.images[0].shape[1]
        sequence.metadata['camera_params'] = read_camera_params()
        sequence.metadata['camera_params'].update({'height': H, 'width' : W})

        # flip tensors
        sequence.images = sequence.images[:, ::-1, ::-1]
        sequence.depths = sequence.depths[:, ::-1, ::-1]
        sequence.imasks = sequence.imasks[:, ::-1, ::-1]
        sequence.poses  = sequence.poses @ np.array([
            [-1,  0,  0,  0],
            [ 0, -1,  0,  0],
            [ 0,  0,  1,  0],
            [ 0,  0,  0,  1],
        ])
        sequence.metadata['camera_params'].update({
            'cx': W - sequence.metadata['camera_params']['cx'],
            'cy': H - sequence.metadata['camera_params']['cy'],
        })
        return sequence
    

if __name__ == '__main__':
    from scenefactor.utils.visualize import visualize_sequence

    reader = GraspNetFrameSequenceReader(base_dir='/home/gtangg12/data/graspnet', name='scene_0000')
    sequence = reader.read(slice=(0, -1, 250))
    print(sequence)
    #print(np.unique(sequence.images))
    #print(np.unique(sequence.depths))
    #print(np.unique(sequence.imasks))

    visualize_sequence(sequence).save('tests/sequence_graspnet.png')