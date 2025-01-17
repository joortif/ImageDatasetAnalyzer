import cv2
from skimage.feature import local_binary_pattern
from tqdm import tqdm
import numpy as np

from datasetanalyzerlib.image_similarity.embeddings.embedding import Embedding
from datasetanalyzerlib.image_similarity.datasets.imagedataset import ImageDataset

class OpenCVLBPEmbedding(Embedding):
    
    def __init__(self, radius: int, num_points: int, method: str="uniform"):
        self.radius = radius
        self.num_points = num_points
        self.method = method

    def generate_embeddings(self, dataset: ImageDataset):
        embeddings = []

        for image in tqdm(dataset, "Generating embeddings..."):
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

            lbp = local_binary_pattern(gray_image, self.num_points, self.radius, self.method)

            hist, _ = np.histogram(
                lbp.ravel(),
                bins=np.arange(0, self.num_points + 3),
                range=(0, self.num_points + 2),
            )
            hist = hist.astype("float") / hist.sum()
            embeddings.append(hist)

        return np.array(embeddings)
            