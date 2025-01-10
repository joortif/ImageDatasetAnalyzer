import os
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image

class Dataset(Dataset):
    def __init__(self, directory: str, processor):
        """
        Args:
            directory (str): Directory containing images.
            processor : Pretrained processor for image preprocessing.
            image_size (int, optional): The size to resize the images. Defaults to 224.
        """
        self.directory = directory
        self.processor = processor
        self.image_files = [f for f in os.listdir(directory) if f.endswith(('jpg', 'png'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.directory, self.image_files[idx])
        
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Error loading image {image_path}: {e}")

        processed = self.processor(images=image, return_tensors="pt")
        
        inputs = processed.get("pixel_values", image).squeeze(0)

        return inputs
    
    def get_image(self, idx):
        """
        Returns the raw image as a Pillow Image object.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            Image: The raw image as a Pillow Image object.
        """
        image_path = os.path.join(self.directory, self.image_files[idx])

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Error loading image {image_path}: {e}")

        return image