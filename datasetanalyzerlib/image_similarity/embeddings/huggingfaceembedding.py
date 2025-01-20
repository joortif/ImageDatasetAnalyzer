from transformers import AutoModel, AutoProcessor
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

from datasetanalyzerlib.image_similarity.datasets.imagedataset import ImageDataset
from datasetanalyzerlib.image_similarity.embeddings.embedding import Embedding


class HuggingFaceEmbedding(Embedding):
    def __init__(self, model_name: str, batch_size: int=8):
        self.model_name = model_name
        self.processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name)
        self.batch_size = batch_size

        print(f"Loaded {self.model_name} from HuggingFace Hub.")

    def _transform_image(self, batch) -> torch.Tensor:
        """
        Transforms a batch of images into the appropriate tensor format for the model.

        Args:
            batch: A list of images to be processed.

        Returns:
            torch.Tensor: Processed tensor ready for model input.
        """
        images = [image.convert("RGB") for image in batch]
        processed = self.processor(images=images, return_tensors="pt")
        inputs = processed.get("pixel_values", images).squeeze(0)
        return inputs

    def generate_embeddings(self, dataset: ImageDataset, device: torch.device = None):
        """
        Generates embeddings for all images in the specified dataset using a HuggingFace model.

        Args: 
            dataset (ImageDataset): Dataset of images to process.
            device (torch.device, optional): Device to use for computation. Defaults to the best available device.

        Returns:
            torch.Tensor: Embeddings generated for all images in the dataset.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device.type == "cuda":
                device_name = torch.cuda.get_device_name(device.index)  
                print(f"Device not detected. Using GPU: {device_name}")
            else:
                print("Device not detected. Using CPU.")
        
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, collate_fn=lambda batch: self._transform_image(batch))
        
        embeddings = []
        self.model.to(device)
        
        for batch in tqdm(dataloader, desc="Generating embeddings..."):
            batch = batch.to(device)

            with torch.no_grad():
                if hasattr(self.model, "vision_model"):
                    outputs = self.model.vision_model(pixel_values=batch).last_hidden_state[:, 0]
                else:
                    outputs = self.model(pixel_values=batch).last_hidden_state[:, 0]

            embeddings.append(outputs.cpu())

        return torch.cat(embeddings)
