import tensorflow as tf

from torch.utils.data import DataLoader

from PIL import Image

from tqdm import tqdm

import torch
import numpy as np

from datasetanalyzerlib.image_similarity.embeddings.embedding import Embedding
from datasetanalyzerlib.image_similarity.datasets.imagedataset import ImageDataset


class TensorflowEmbedding(Embedding):

    def __init__(self, model_name: str, batch_size: int=8, resize_height: int=224, resize_width: int=224):

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

    def _load_model(self): 
        try:
            model_name_lower = self.model_name.lower()

            versions = ["v1", "v2", "v3"]

            for version in versions:
                if version in model_name_lower:
                    model_name_lower = model_name_lower.replace(version, f"_{version}")

            model_module = getattr(tf.keras.applications, model_name_lower)

            model_class = getattr(model_module, self.model_name)

            model = model_class(weights='imagenet', input_shape=(self.height, self.width, 3), include_top=False)
            
            preprocess_input = model_module.preprocess_input
            return model, preprocess_input
        except AttributeError:
            raise ValueError(f"Model {self.model_name} not supported or not found in tensorflow.keras.applications.")
        
    def _transform_image(self, batch) -> torch.Tensor:
        resized_batch = []
        for image in batch:
            image = image.resize((self.width, self.height))
            image = np.array(image)

            image = image / 255.0  
            image = (image - self.mean) / self.std

            resized_batch.append(image)
        
        return torch.stack([torch.tensor(b, dtype=torch.float32) for b in resized_batch])
        
    def generate_embeddings(self, dataset: ImageDataset, device: str=None):        
        if device is None:
            if tf.config.list_physical_devices('GPU'):
                print("Device detected. Using GPU.")
            else:
                print("Device not detected. Using CPU.")

        elif device == "GPU" and tf.config.list_physical_devices('GPU'):
            print("Using GPU as specified.")
        else:
            print("Device not detected or specified. Using CPU.")  
        
        
        dataset.set_processor(self.processor)

        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, collate_fn=lambda batch: self._transform_image(batch))

        embeddings = []

        for batch in tqdm(dataloader, desc="Generando embeddings..."):
            
            batch_tensor = tf.convert_to_tensor(batch, dtype=tf.float32)
            features = self.model(batch_tensor, training=False)

            global_max_pool = tf.reduce_max(features, axis=(1, 2))  
        
            embeddings.append(global_max_pool.numpy())

        return np.vstack(embeddings)