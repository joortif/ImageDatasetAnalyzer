import numpy as np
import torch

from imagedatasetanalyzer.src.datasets.imagedataset import ImageDataset

class Embedding:

    def generate_embeddings(self, dataset: ImageDataset) -> np.ndarray:
        pass

    def _transform_image(self, batch) -> torch.Tensor:
        pass