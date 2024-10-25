import numpy as np

from scenefactor.data.common import NumpyTensor
from scenefactor.data.sequence import FrameSequence
from scenefactor.utils.geom import dialate_bmask


SEMANTIC_BACKGROUND = 0
INSTANCE_BACKGROUND = 0