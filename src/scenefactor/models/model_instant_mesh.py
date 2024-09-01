from omegaconf import OmegaConf

from scenefactor.data.common import NumpyTensor


class ModelInstantMesh:
    """
    """
    def __init__(self, config: OmegaConf):
        """
        """
        pass

    def __call__(self, image: NumpyTensor['n', 'h', 'w', 3]):
        """
        """
        # make cache
        # for each image in batch process image
        # delete cache
        pass

    def process(self, image: NumpyTensor['h', 'w', 3]):
        """
        """
        # save image crop as filename to cache
        # run script provided by instantMesh with specified save dir
        # load mesh/video from save dir and write to permanent location