from torchvision import models, transforms
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasetanalyzerlib.image_similarity.embeddings.embedding import Embedding
from datasetanalyzerlib.image_similarity.datasets.imagedataset import ImageDataset

class PyTorchEmbedding(Embedding):

    def __init__(self, model_name: str, batch_size: int=8):
        self.weights = models.get_model_weights(model_name).DEFAULT
        self.processor = self.weights.transforms()

        self.model = models.get_model(model_name)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])

        self.batch_size = batch_size
        print(f"Loaded {model_name} from PyTorch.")

    def _transform_image(self, batch) -> torch.Tensor:
        """
        Transforms a batch of images into the appropriate tensor format for the model.

        Args:
            batch: A list of images to be processed.

        Returns:
            torch.Tensor: Processed tensor ready for model input.
        """
        images = [self.processor(image.convert("RGB")) for image in batch]
        return torch.stack(images)

    def generate_embeddings(self, dataset: ImageDataset, device: torch.device = None):
        """
        Generates embeddings for all images in the specified dataset using a PyTorch model.

        Args:
            dataset (ImageDataset): Dataset of images to process. The dataset is expected to be compatible 
                                    with PyTorch DataLoader and should support setting a processor.
            device (torch.device, optional): Device to use for computation. Defaults to the best available device.

        Returns:
            torch.Tensor: A tensor containing the embeddings for all images in the dataset.
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
                outputs = self.model(batch)

                if len(outputs.shape) == 4:
                    outputs = outputs.mean(dim=[2, 3])

                embeddings.append(outputs.squeeze().cpu())
            
        return torch.cat(embeddings)
        