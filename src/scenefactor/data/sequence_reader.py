from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from scenefactor.data.common import NumpyTensor, TorchTensor
from scenefactor.data.sequence import FrameSequence


class FrameSequenceReader(ABC):
    """
    """
    def __init__(self):
        """
        """
        pass
    
    @abstractmethod
    def read(self, filename: Path | str) -> FrameSequence:
        """
        """
        pass

    def __len__(self):
        pass
        

class ScanNetPPFrameSequenceReader(FrameSequenceReader):
    """
    """
    def read(self, filename: Path | str) -> FrameSequence:
        """
        """
        pass