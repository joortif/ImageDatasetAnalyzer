import tensorflow as tf

from torch.utils.data import DataLoader

from tqdm import tqdm

import re
import torch
import numpy as np

from imagedatasetanalyzer.src.embeddings.embedding import Embedding
from imagedatasetanalyzer.src.datasets.imagedataset import ImageDataset


class TensorflowEmbedding(Embedding):
    """
    TensorflowEmbedding class for generating embeddings using pre-trained TensorFlow models. ImageNet weights are used
    and its normalization attributes (pixel mean and std).

    This class utilizes TensorFlow's pre-trained models to extract feature embeddings from image datasets.
    The embeddings can be used for tasks such as clustering, classification, or visualization.

    Attributes:
        model_name (str): The name of the pre-trained model to use from TensorFlow.
        batch_size (int): The number of images to process in each batch.
        resize_height (int): The height to resize images for the model.
        resize_width (int): The width to resize images for the model.
    """

    def __init__(self, model_name: str, batch_size: int=8, resize_height: int=224, resize_width: int=224):
        """
        Args:
            model_name (str): The name of the pre-trained model to use from TensorFlow.
            batch_size (int, optional): The number of images to process in each batch. Defaults to 8.
            resize_height (int, optional): The height to resize images for the model. Defaults to 224.
            resize_width (int, optional): The width to resize images for the model. Defaults to 224.
        """
        self.model_name = model_name
        self.height = resize_height
        self.width = resize_width

        self.model, self.processor = self._load_model()
        self.batch_size = batch_size
        self.model.trainable = False  
        self.model = tf.keras.Model(inputs=self.model.input, outputs=self.model.layers[-2].output)

        self.mean = np.array([0.485, 0.456, 0.406])  
        self.std = np.array([0.229, 0.224, 0.225]) 

        print(f"Loaded {self.model_name} from TensorFlow.")

    def _clean_model_name(self) -> str:
        """
        Cleans and returns a simplified version of the model name by removing certain version identifiers, 
        size descriptors, and digits that are not part of the model's name.

        Returns:
        str: The cleaned and simplified model name, with sizes and unnecessary numbers removed.
        """
        model_name = self.model_name.lower()
        versions = ["v2", "v3"]
        sizes = ["tiny", "small", "base", "xlarge", "large" , "mobile"]

        for size in sizes:
            if model_name.endswith(size):
                model_name = model_name.replace(size, '')

        name_without_digits = re.sub(r'(?<!v)\d+', '', model_name)

        for version in versions:
            if version in name_without_digits:
                name_without_digits = name_without_digits.replace(version, f"_{version}")

        return name_without_digits.strip()


    def _load_model(self): 
        """
        Loads the TensorFlow model and its associated preprocessing function.

        Returns:
            tuple:
                - tf.keras.Model: A pre-trained TensorFlow model configured for feature extraction.
                - function: A preprocessing function for input images.

        Raises:
            ValueError: If the specified model name is not supported or not found in `tensorflow.keras.applications`.
        """
        try:
            model_name_lower = self._clean_model_name()

            model_module = getattr(tf.keras.applications, model_name_lower)

            model_class = getattr(model_module, self.model_name)

            model = model_class(weights='imagenet', input_shape=(self.height, self.width, 3), include_top=False)
            
            preprocess_input = model_module.preprocess_input
            return model, preprocess_input
        except AttributeError:
            raise ValueError(f"Model {self.model_name} not supported or not found in tensorflow.keras.applications.")
        
    def _transform_image(self, batch) -> torch.Tensor:
        """
        Resizes and normalizes a batch of images for input into the TensorFlow model.

        Args:
            batch (list): A list of PIL.Image objects to process.

        Returns:
            torch.Tensor: A tensor containing the preprocessed batch of images.
                        Each image is resized, normalized, and converted to a float tensor.
        """
        resized_batch = []
        for image in batch:
            image = image.resize((self.width, self.height))
            image = np.array(image)

            image = self.processor(image)

            image = image.copy()

            resized_batch.append(image)
        
        return tf.convert_to_tensor(np.array(resized_batch), dtype=tf.float32)
        
    def generate_embeddings(self, dataset: ImageDataset, device: str=None):        
        """
        Generates embeddings for all images in the specified dataset using a TensorFlow model.

        Args:
            dataset (ImageDataset): Dataset of images to process. The dataset is expected to be compatible
                                    with PyTorch DataLoader and should support setting a processor.
            device (str, optional): Device to use for computation. If "GPU" is specified and a GPU is available, 
                                    it will be used. Defaults to CPU if not specified or if GPU is unavailable.

        Returns:
            np.ndarray: A NumPy array containing the embeddings for all images in the dataset.
        """
        
        if device is None:
            if tf.config.list_physical_devices('GPU'):
                print("Device detected. Using GPU.")
            else:
                print("Device not detected. Using CPU.")

        elif device == "GPU" and tf.config.list_physical_devices('GPU'):
            print("Using GPU as specified.")
        else:
            print("Device not detected or specified. Using CPU.")  
        
        
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, collate_fn=lambda batch: self._transform_image(batch))

        embeddings = []

        for batch in tqdm(dataloader, desc="Generando embeddings..."):
            
            batch_tensor = tf.convert_to_tensor(batch, dtype=tf.float32)
            features = self.model(batch_tensor, training=False)

            if batch_tensor.shape[-1] != 3:  
                raise ValueError("Images must have 3 channels (RGB) in channels_last format.")

            features = self.model(batch_tensor, training=False)
            
            if len(features.shape) == 4: 

                global_max_pool = tf.reduce_max(features, axis=(1, 2)) 
                embeddings.append(global_max_pool.numpy())
            else:
                raise ValueError(f"Expected 4D tensor (batch_size, height, width, channels), but got shape: {features.shape}")

        return np.vstack(embeddings)