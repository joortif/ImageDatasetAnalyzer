from transformers import AutoModel, AutoProcessor
from torchvision import transforms as T
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

from .dataset import Dataset

class Embedding:
    def __init__(self, model_name: str, batch_size: int=8):
        """
        Creates embedding class with a pretrained model from Hugging Face.

        Args: 
            model (str): Pretrained model's name or route.
        """
        self.model_name = model_name
        self.processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name)
        self.batch_size = batch_size

    def generate_embedding(self, directory: str, device: torch.device = None):
        """
        Processes all images in the specified directory in batches.

        Args: 
            directory (str): Directory containing images.
            device (torch.device, optional): Device to use for computation. Defaults to the best available device.

        Returns:
            torch.Tensor: Embeddings generated for all images in the directory.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device.type == "cuda":
                device_name = torch.cuda.get_device_name(device.index)  
                print(f"Device not detected. Using GPU: {device_name}")
            else:
                print("Device not detected. Using CPU.")
        
        dataset = Dataset(directory, self.processor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
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