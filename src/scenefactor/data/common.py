from pathlib import Path

from torchtyping import TensorType


NumpyTensor = TensorType
TorchTensor = TensorType


CONFIGS_DIR = Path(__file__).resolve().parent / '../../../configs'