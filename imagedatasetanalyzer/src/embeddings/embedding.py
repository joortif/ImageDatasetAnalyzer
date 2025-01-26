import numpy as np
import torch

from imagedatasetanalyzer.src.datasets.imagedataset import ImageDataset

class Embedding:
    """
    Represents an embedding generator for image datasets.

    Methods:
        generate_embeddings(dataset: ImageDataset) -> np.ndarray:
            Generates embeddings for a given image dataset.

        _transform_image(batch) -> torch.Tensor:
            Transforms a batch of images into a format suitable for embedding generation.
    """
    def generate_embeddings(self, dataset: ImageDataset) -> np.ndarray:
        pass

    def _transform_image(self, batch) -> torch.Tensor:
        pass