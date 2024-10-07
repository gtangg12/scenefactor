import json
from pathlib import Path
from glob import glob
from natsort import natsorted

import numpy as np

from scenefactor.data.common import NumpyTensor
from scenefactor.data.sequence import FrameSequence
from scenefactor.data.sequence_reader_base import FrameSequenceReader


PATH = Path(__file__).parent


class ReplicaVMapFrameSequenceReader(FrameSequenceReader):
    """
    """
    READER_CONFIG = PATH / 'sequence_reader_replica_vmap.yaml'

    def __init__(
        self, 
        base_dir: Path | str, 
        save_dir: Path | str, name: str, track='01'
    ):
        """
        """
        super().__init__(base_dir, save_dir, name)

        assert track in ['00', '01']
        self.track = track
        self.data_dir = self.data_dir / 'imap' / track
        self.save_dir = self.save_dir / track

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
            poses = np.loadtxt(self.data_dir / 'traj_w_c.txt', delimiter=' ')
            poses = poses.reshape(-1, 4, 4)
            poses = poses[slice[0]:slice[1]:slice[2]]
            poses = poses @ np.array(self.metadata['pose_axis_transform'])
            return poses
        
        def read_semantic_info() -> dict[int, dict]:
            """
            """
            things = self.metadata.pop('semantic_classes_things')
            with open(self.base_dir / self.name / 'habitat/info_preseg_semantic.json', 'r') as f:
                semantic_info = json.load(f)['classes']
            semantic_info = {
                data['id']: {'name': data['name'], 'class': 'thing' if data['name'] in things else 'stuff'}
                for data in semantic_info
            }
            unknown = max(semantic_info.keys()) + 1
            semantic_info[unknown] = {'name': 'unknown', 'class': 'thing'}
            return semantic_info, unknown
        
        image_filenames = read_filenames('rgb/*.png')
        depth_filenames = read_filenames('depth/*.png')
        smask_filenames = read_filenames('semantic_class/semantic_class_*.png')
        imask_filenames = read_filenames('semantic_instance/semantic_instance_*.png')
        
        poses = read_poses()

        self.metadata['semantic_info'], semantic_label_unknown = read_semantic_info()
        
        sequence = FrameSequence(
            poses=poses,
            images=np.array([self.load_image(f, resize) for f in image_filenames]),
            depths=np.array([self.load_depth(f, resize, scale=self.metadata['depth_scale']) for f in depth_filenames]),
            smasks=np.array([self.load_smask(f, resize) for f in smask_filenames]),
            imasks=np.array([self.load_imask(f, resize) for f in imask_filenames]),
            metadata=self.metadata
        )
        sequence.smasks[sequence.smasks == 0] = semantic_label_unknown # replica labels unknown semantic class as 0
        return sequence
    

if __name__ == '__main__':
    from scenefactor.data.sequence import save_sequence, load_sequence

    reader = ReplicaVMapFrameSequenceReader('/home/gtangg12/data/replica-vmap/office_0')
    sequence = reader.read()
    print(sequence)

    save_sequence('tests/sequence', sequence)
    sequence = load_sequence('tests/sequence')
    print(sequence)

    sequence_item = sequence[0]
    print(sequence_item)

    sequence_slice = sequence[0:10:2]
    print(sequence_slice)
    print(sequence)

    sequence2 = sequence.clone()
    sequence = None
    print(sequence)
    print(sequence2)