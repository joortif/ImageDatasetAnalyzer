from typing import Optional

import cv2
from skimage.feature import local_binary_pattern
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
import torch

from imagedatasetanalyzer.embeddings.embedding import Embedding
from imagedatasetanalyzer.datasets.imagedataset import ImageDataset

class OpenCVLBPEmbedding(Embedding):
    """
    OpenCVLBPEmbedding class for generating Local Binary Pattern (LBP) embeddings using OpenCV.

    This class uses OpenCV to compute LBP features from images, which are then converted into embeddings.
    These embeddings can be used for various tasks like clustering, classification, or visualization.

    Attributes:
        radius (int): The radius of the LBP neighborhood.
        num_points (int): The number of points to consider in the LBP calculation.
        batch_size (int): The number of images to process in each batch.
        method (str): The LBP method (e.g., 'uniform').
        resize_height (int | None): The height to resize images to (optional).
        resize_width (int | None): The width to resize images to (optional).
    """    
    def __init__(self, radius: int, num_points: int, resize_height:  Optional[int] = None, resize_width: Optional[int] = None, batch_size: int = 8, method: str="uniform"):
        """
        Args:
            radius (int): The radius of the LBP neighborhood.
            num_points (int): The number of points to consider in the LBP calculation.
            resize_height (int | None, optional): The height to resize images to (optional).
            resize_width (int | None, optional): The width to resize images to (optional).
            batch_size (int, optional): The number of images to process in each batch. Defaults to 8.
            method (str, optional): The LBP method to use (e.g., 'uniform'). Defaults to 'uniform'.
        """
        self.radius = radius
        self.num_points = num_points
        self.batch_size = batch_size
        self.method = method
        self.resize_height = resize_height
        self.resize_width = resize_width

    def _transform_image(self, batch) -> torch.Tensor:
        """
        Transforms a batch of images into the appropriate tensor format for the model.

        Args:
            batch: A list of images to be processed.

        Returns:
            torch.Tensor: Processed tensor ready for model input.
        """        
        images = [np.array(image.convert("RGB")) for image in batch]        
        if self.resize_height and self.resize_width:
            images = [cv2.resize(image, (self.resize_width, self.resize_height)) for image in images]

        images = np.stack(images)  
        gray_images = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images])

        return gray_images

    def generate_embeddings(self, dataset: ImageDataset) -> np.ndarray:
        """
        Generates embeddings for the images in the given dataset using Local Binary Patterns (LBP) 
        for feature extraction.

        Args:
            ImageDataset: Dataset of images to generate embeddings for.

        Returns:
            np.ndarray: NumPy array where each row corresponds to the LBP-based histogram embedding of an image 
            in the dataset.
        """
        embeddings = []

        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, collate_fn=lambda batch: self._transform_image(batch))
        for batch in tqdm(dataloader, "Generating embeddings..."):
            for gray_image in batch:
                lbp = local_binary_pattern(gray_image, self.num_points, self.radius, self.method)

                hist, _ = np.histogram(
                    lbp.ravel(),
                    bins=np.arange(0, self.num_points + 3),
                    range=(0, self.num_points + 2),
                )
                hist = hist.astype("float") / hist.sum()
                embeddings.append(hist)

        return np.array(embeddings)