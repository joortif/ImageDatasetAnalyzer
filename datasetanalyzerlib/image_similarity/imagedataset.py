import os
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, directory: str, image_files: np.ndarray=None, processor=None):
        """
        Args:
            directory (str): Directory containing images.
            processor(optional) : Pretrained processor for image preprocessing.
            image_files (array, optional): Images to save from the directory. If None, all the images from the directory are saved.
        """

        self.directory = directory
        self.processor = processor

        self.image_files = image_files
        
        if not self.image_files:
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

    def set_processor(self, processor) -> None:
        """
        Assigns a processor to the dataset.

        This method sets the processor responsible for handling the images 
        within the dataset. It allows for preprocessing or transformations 
        to be applied uniformly.

        Args:
            processor: Processor for images of the dataset.

        Returns:
            None: This method does not return a value.

        """

        self.processor = processor


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