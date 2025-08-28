import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

from collections import defaultdict
import logging
from tqdm import tqdm

from imagedatasetanalyzer.utils.metrics import compute_LPIPS, compute_SSIM

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

        self.logger = logging.getLogger(self.__class__.__name__)

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
            raise RuntimeError(f"Error loading image {image_path}: {e}") from e

        return image

    def image_sizes(self): 
        """
        Returns the sizes of the images in the directory.
        Also calculates the average width and height.
        """
        images_sizes = defaultdict(int)
        total_width = 0
        total_height = 0

        for fname in tqdm(self.image_files, desc="Reading files"):
            fpath = os.path.join(self.img_dir, fname)
            with Image.open(fpath) as img:
                width, height = img.size
                total_width += width
                total_height += height
                images_sizes[(width, height)] += 1

        mode_height, mode_width = max(set(images_sizes), key=list(images_sizes.values()).count)
        sorted_sizes = sorted(images_sizes.items(), key=lambda item: item[1], reverse=True)
        images_sizes = dict(sorted_sizes)
        
        for size, count in images_sizes.items():
            width, height = size
            percentage = (count / len(self.image_files)) * 100
            self.logger.info(f"Size {width}x{height}: {count} images ({percentage:.2f}%)")

        avg_width = round(total_width / len(self.image_files))
        avg_height = round(total_height / len(self.image_files))
        self.logger.info(f"Average image size: {avg_height}x{avg_width}")
        self.logger.info(f"Image size mode: {mode_height}x{mode_width}")

        return mode_height, mode_width

    def dataset_similarity(self, similarity_index, logger):

        if not similarity_index:
            return
        
        similarity_index = [idx.strip().upper() for idx in similarity_index]
        valid_indices = {"SSIM", "LPIPS"}
        selected_indices = [idx for idx in similarity_index if idx in valid_indices]

        if not selected_indices:
            logger.warning("No valid similarity index selected. Supported options: SSIM, LPIPS.")
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if device.type == "cpu":
            logger.warning("Similarity analysis using SSIM or LPIPS can be computationally expensive on CPU. Please use a machine with a CUDA-compatible GPU.")
            return
            
        logger.info("Computing similarity indices for the dataset")
        for index in selected_indices:
            if index == "SSIM":
                ssim_mean, ssim_std, _ = compute_SSIM(self.img_dir, device)
                logger.info("Dataset SSIM Index: %.4f, Std Dev: %.4f", ssim_mean, ssim_std)
            elif index == "LPIPS":
                lpips_mean, lpips_std, _ = compute_LPIPS(self.img_dir, device)
                logger.info("Dataset LPIPS Index: %.4f, Std Dev: %.4f", lpips_mean, lpips_std)

    
    def analyze(self, similarity_index=["SSIM","LPIPS"], verbose=False, log_dir=None):
        """
        Analyzes the image dataset reporting the distribution of image sizes.

        This method calculates the frequency of each unique image size in the dataset
        and prints the report to the console. It also calculates SSIM or LPIPS similarity indexes
        from all the dataset optionally.
        """
        
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
        self.image_sizes(self.logger)
        self.logger.info("Total number of images in the dataset: %s", len(self.image_files))

        self.dataset_similarity(similarity_index, self.logger)