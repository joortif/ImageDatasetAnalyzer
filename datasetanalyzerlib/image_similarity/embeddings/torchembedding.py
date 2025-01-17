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

    def generate_embeddings(self, dataset: ImageDataset, device: torch.device = None):

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device.type == "cuda":
                device_name = torch.cuda.get_device_name(device.index)  
                print(f"Device not detected. Using GPU: {device_name}")
            else:
                print("Device not detected. Using CPU.")
        
        dataset.set_processor(self.processor)
        
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        embeddings = []

        self.model.to(device)
        
        for batch in tqdm(dataloader, desc="Generating embeddings..."):
            batch = batch.to(device)

            with torch.no_grad():
                outputs = self.model(batch)

                embeddings.append(outputs.squeeze().cpu())
            
        return torch.cat(embeddings)
        