from collections import defaultdict
import os
import logging
from tqdm import tqdm

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
    def __init__(self, img_dir: str, image_files: np.ndarray=None):
        """
        Args:
            directory (str): Directory containing images.
            image_files (array, optional): Images to save from the directory. If None, all the images from the directory are saved.
        """

        self.img_dir = img_dir

        self.image_files = image_files
        
        if not self.image_files:
            self.image_files = [f for f in os.listdir(img_dir) if f.endswith(('jpg', 'png'))]

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


    def _image_sizes(self, directory, files, logger): 
        """
        Returns the sizes of the images in the directory.
        """
        images_sizes = defaultdict(int)
        for fname in tqdm(files, desc="Reading files"):
            fpath = os.path.join(directory, fname)
            with Image.open(fpath) as img:
                size = img.size
                images_sizes[size] += 1

        sorted_sizes = sorted(images_sizes.items(), key=lambda item: item[1], reverse=True)

        images_sizes = dict(sorted_sizes)
        
        for size, count in images_sizes.items():
            width, height = size
            percentage = (count / len(files)) * 100
            logger.info(f"Size {width}x{height}: {count} images ({percentage:.2f}%)")
    
    def analyze(self, verbose=False, log_dir=None):
        """
        Analyzes the image dataset reporting the distribution of image sizes.

        This method calculates the frequency of each unique image size in the dataset
        and prints the report to the console.
        """
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        if not self.logger.hasHandlers():
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')

            if not log_dir:
                log_dir = os.getcwd()

            file_handler = logging.FileHandler(os.path.join(log_dir, "logs.txt"), mode='w')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
                
            if verbose:
                stream_handler = logging.StreamHandler()
                stream_handler.setFormatter(formatter)
                self.logger.addHandler(stream_handler)

            self.logger.setLevel(logging.INFO)

        self.logger.info("Calculating image sizes...")
        self._image_sizes(self.img_dir, self.image_files, self.logger)
        self.logger.info(f"Total number of images in the dataset: {len(self.image_files)}")
