import re
from collections import deque

from omegaconf import OmegaConf

from scenefactor.data.common import NumpyTensor, TorchTensor
from scenefactor.models.model_gpt import ModelGpt, ModelGptInput, unpack_content
from scenefactor.utils.geom import *
from scenefactor.utils.tensor import *


class RankingLoss:
    """
    """
    def __init__(self, config: OmegaConf):
        """
        """
        self.config = config


class OccupancyRankingLoss(RankingLoss):
    """
    """
    def __call__(self, data: dict) -> float:
        """
        """
        return bbox_area(data['bbox']) / data['bmask'].sum()


VLM_RANKING_PROMPT = """You have several images (view_0 to view_n) of an object from different angles. Some views may show the object partially, from odd angles, or in low quality. Choose the best view to input to an image to 3D generative model.

Output "The best view is view_i. Explanation:..." (replacing i with the index) and explain briefly. Please adhere exactly to the aforementioned format, since we parse the view index using regex.

In your explanation, first determine the identiy of the object based on all views. Then evaluate based on:
1. Focus on views where the object is easily recognizable and the 3D structure is apparent.
2. Avoid views with occlusion or partial visibility.
3. Avoid views where the object is misaligned, tilted, or appears distorted due to the camera view.
4. Avoid views with poor quality or lighting.

If all the views contain partial or occluded objects, then output "The best view doesn't exist". Most of the time, there is a clear best view."""


class VLMRankingLoss(RankingLoss):
    """
    """
    def __init__(self, config: OmegaConf):
        """
        """
        super().__init__(config)
        self.model = ModelGpt(model=self.config.model)
    
    def __call__(self, images: list[NumpyTensor['h', 'w', 3]]) -> int:
        """
        """
        BATCH_SIZE = 6

        queue = deque()
        for index, image in enumerate(images):
            queue.append((image, index))
        
        while len(queue) > 1:
            batch = []
            while len(batch) < BATCH_SIZE and len(queue) > 0:
                batch.append(queue.popleft())
            batch_index = self.process([image for image, _ in batch])
            if batch_index is None:
                continue
            index = batch[batch_index][1]
            queue.append((images[index], index))
        return None if len(queue) == 0 else queue.pop()[1]
    
    def process(self, images: list[NumpyTensor['h', 'w', 3]]) -> int:
        """
        """
        self.model.reset() # clear previous context
       
        if len(images) == 1:
            return 0
        input = ModelGptInput()
        input.append(VLM_RANKING_PROMPT)
        input.extend([Image.fromarray(image) for image in images])
        response = self.model(input, temperature=1e-6) # see OpenAI documentation on temperature = 0
        response = unpack_content(response)
        #print()
        #print(response)
        match = re.search('The best view is view_(\d+)', response)
        if match:
            index = int(match.group(1))
            if index < 0 or index >= len(images):
                return None
            return index
        return None
    

if __name__ == '__main__':
    module = VLMRankingLoss(OmegaConf.create({'model': 'gpt-4o'}))
    filenames = [
        '/home/gtangg12/scenefactor/tests/instant_mesh_view_0.png',
        '/home/gtangg12/scenefactor/tests/instant_mesh_view_1.png',
        '/home/gtangg12/scenefactor/tests/instant_mesh_view_2.png',
        '/home/gtangg12/scenefactor/tests/instant_mesh_view_3.png',
        '/home/gtangg12/scenefactor/tests/instant_mesh_view_4.png',
        '/home/gtangg12/scenefactor/tests/instant_mesh_view_5.png',
    ]
    images = [read_image(filename) for filename in filenames] # view 0, 5 best
    print(module(images))
    images = [images[1], images[2], images[4], images[3], images[0], images[5]] # view 4, 5 best
    print(module(images))
    
    filenames = [
        '/home/gtangg12/scenefactor/outputs_factorization_old/extractor/visualizations/iteration_0/label_13_index_11.png',
        '/home/gtangg12/scenefactor/outputs_factorization_old/extractor/visualizations/iteration_0/label_13_index_12.png',
    ]
    images = [read_image(filename) for filename in filenames] # view 0 best
    print(module(images))
    images = [images[1], images[0]] # view 1 best
    print(module(images))

    filenames = [
        '/home/gtangg12/scenefactor/outputs_factorization_old/extractor/visualizations/iteration_0/label_79_index_6.png',
        '/home/gtangg12/scenefactor/outputs_factorization_old/extractor/visualizations/iteration_0/label_79_index_7.png',
        '/home/gtangg12/scenefactor/outputs_factorization_old/extractor/visualizations/iteration_0/label_79_index_8.png',
        '/home/gtangg12/scenefactor/outputs_factorization_old/extractor/visualizations/iteration_0/label_79_index_9.png',
        '/home/gtangg12/scenefactor/outputs_factorization_old/extractor/visualizations/iteration_0/label_79_index_10.png',
        '/home/gtangg12/scenefactor/outputs_factorization_old/extractor/visualizations/iteration_0/label_79_index_18.png',
        '/home/gtangg12/scenefactor/outputs_factorization_old/extractor/visualizations/iteration_0/label_79_index_19.png',
    ]
    images = [read_image(filename) for filename in filenames] # anything but 3 4 5
    print(module(images))
    images = [images[1], images[2], images[3], images[4], images[5], images[6], images[0]] # anything but 2 3 4
    print(module(images))

    filenames = [
        '/home/gtangg12/scenefactor/outputs_factorization_old/extractor/visualizations/iteration_0/label_75_index_11.png',
        '/home/gtangg12/scenefactor/outputs_factorization_old/extractor/visualizations/iteration_0/label_75_index_18.png',
        '/home/gtangg12/scenefactor/outputs_factorization_old/extractor/visualizations/iteration_0/label_75_index_19.png',
    ]
    images = [read_image(filename) for filename in filenames] # view 0 best
    print(module(images))
    images = [images[1], images[2], images[0]] # view 2 best
    print(module(images))