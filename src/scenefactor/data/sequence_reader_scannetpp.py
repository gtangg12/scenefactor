from pathlib import Path

from scenefactor.data.sequence import FrameSequence
from scenefactor.data.sequence_reader_base import FrameSequenceReader


PATH = Path(__file__).parent


class ScanNetPPFrameSequenceReader(FrameSequenceReader):
    """
    """
    READER_CONFIG = PATH / 'sequence_reader_scannetpp.yaml'

    def __init__(
        self, 
        base_dir: Path | str, 
        save_dir: Path | str, name: str
    ):
        """
        """
        super().__init__(base_dir, save_dir, name)

        pass

    def read(self, slice=(0, -1, 20), resize: tuple[int, int]=None, override=False) -> FrameSequence:
        """
        """
        pass