import base64
from io import BytesIO

from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from MedImageInsights.Models.medimageinsightmodel import MedImageInsight
from imagedatasetanalyzer.embeddings.embedding import Embedding


class MedImageInsightEmbedding(Embedding):
    """
    MedImageInsightEmbedding class for generating image embeddings using the Microsoft model MedImageInsight.

    This class uses the library MedImageInsights to use the model for extracting feature embeddings from 
    images.
    """
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.model = MedImageInsight()
        self.model.load_model()

    def read_image_bytes(self, path: str) -> bytes:
        with open(path, 'rb') as f:
            return f.read()

    def image_to_base64(self, img: Image.Image) -> str:
        buffer = BytesIO()
        img.save(buffer, format=img.format)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _transform_image(self, batch):
        """
        Transforms a batch of images to base64 for the processing of embeddings.

        Args:
            batch: A list of images to be processed.

        Returns:
            torch.Tensor: Processed tensor ready for model input.
        """
        images_b64 = [self.image_to_base64(img) for img in batch]
        return images_b64

    def generate_embeddings(self, dataset):

        embeddings_dict = {}
        start_idx=0

        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=None, collate_fn=lambda batch: self._transform_image(batch))

        for batch in tqdm(dataloader, "Generating embeddings..."):

            image_embeddings = self.model.encode(images=batch)["image_embeddings"]

            batch_filenames = dataset.image_files[start_idx:start_idx + len(batch)]

            for file, emb in zip(batch_filenames, image_embeddings):
                embeddings_dict[file] = emb

            start_idx+=len(batch)

        return embeddings_dict