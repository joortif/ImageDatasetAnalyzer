from collections import defaultdict
import os

from torch.utils.data import Dataset

import numpy as np
from PIL import Image


class ImageDataset(Dataset):
    """
    Represents a dataset of images stored in a directory.

    This class provides functionality to load images, retrieve individual images,
    and analyze the distribution of image sizes in the dataset.

    Attributes:
        img_dir (str): Path to the directory containing the images.
        image_files (np.ndarray): List of image filenames in the directory. If not provided, all images in the directory will be included.
    """
    def __init__(self, image_dir: str, image_files: np.ndarray=None):
        """
        Args:
            directory (str): Directory containing images.
            image_files (array, optional): Images to save from the directory. If None, all the images from the directory are saved.
        """

        self.img_dir = image_dir

        self.image_files = image_files
        
        if not self.image_files:
            self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('jpg', 'png'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.img_dir, self.image_files[idx])
        image = Image.open(image_path)

        return image

    def get_image(self, idx):
        """
        Returns the raw image as a Pillow Image object.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            Image: The raw image as a Pillow Image object.
        """
        image_path = os.path.join(self.img_dir, self.image_files[idx])

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Error loading image {image_path}: {e}")

        return image


    def _image_sizes(self, directory, files): 
        """
        Returns the sizes of the images in the directory.
        """
        images_sizes = defaultdict(int)
        for fname in files:
            fpath = os.path.join(directory, fname)
            with Image.open(fpath) as img:
                size = img.size
                images_sizes[size] += 1

        sorted_sizes = sorted(images_sizes.items(), key=lambda item: item[1], reverse=True)

        images_sizes = dict(sorted_sizes)
        
        for size, count in images_sizes.items():
            width, height = size
            percentage = (count / len(files)) * 100
            print(f"Size {width}x{height}: {count} images ({percentage:.2f}%)")
    
    def analyze(self):
        """
        Analyzes the image dataset reporting the distribution of image sizes.

        This method calculates the frequency of each unique image size in the dataset
        and prints the report to the console.
        """
        
        self._image_sizes(self.img_dir, self.image_files)
        print(f"Total number of images in the dataset: {len(self.image_files)}")
