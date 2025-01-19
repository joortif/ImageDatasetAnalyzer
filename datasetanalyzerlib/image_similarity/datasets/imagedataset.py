from collections import defaultdict
import os

from torch.utils.data import Dataset
import torch

import numpy as np
from PIL import Image

import tensorflow as tf

class ImageDataset(Dataset):
    def __init__(self, image_dir: str, image_files: np.ndarray=None, processor=None):
        """
        Args:
            directory (str): Directory containing images.
            processor(optional) : Pretrained processor for image preprocessing.
            image_files (array, optional): Images to save from the directory. If None, all the images from the directory are saved.
        """

        self.img_dir = image_dir
        self.processor = processor

        self.image_files = image_files
        
        if not self.image_files:
            self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('jpg', 'png'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.img_dir, self.image_files[idx])
        image = Image.open(image_path)

        return image

        if self.processor is None:
            return np.array(image)
        
        image = image.convert("RGB")

        if hasattr(self.processor, '__call__'):
            if isinstance(self.processor, torch.nn.Module):  
                transform = self.processor
                return transform(image)
            try:
                processed = self.processor(images=image, return_tensors="pt")
                inputs = processed.get("pixel_values", image).squeeze(0)

                return inputs
            except Exception as e:
                print(self.processor.type)
                image_np = np.array(image).copy()
                processed = self.processor(image_np)
                return processed
            
        transform = self.processor

        return transform(image)

        

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
