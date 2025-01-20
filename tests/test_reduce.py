import sys
import os
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'datasetanalyzerlib')))

from datasetanalyzerlib.image_similarity.embeddings.embedding import Embedding
from datasetanalyzerlib.image_similarity.models.kmeansclustering import KMeansClustering
from datasetanalyzerlib.image_similarity.models.agglomerativeclustering import AgglomerativeClustering
from datasetanalyzerlib.image_similarity.models.dbscanclustering import DBSCANClustering
from datasetanalyzerlib.image_similarity.models.opticsclustering import OPTICSClustering
from datasetanalyzerlib.image_similarity.datasets.imagedataset import ImageDataset
from datasetanalyzerlib.image_similarity.embeddings.huggingfaceembedding import HuggingFaceEmbedding
from datasetanalyzerlib.image_similarity.datasets.imagelabeldataset import ImageLabelDataset
from datasetanalyzerlib.image_similarity.embeddings.opencvlbpembedding import OpenCVLBPEmbedding
from datasetanalyzerlib.image_similarity.embeddings.torchembedding import PyTorchEmbedding
from datasetanalyzerlib.image_similarity.embeddings.tensorflowembedding import TensorflowEmbedding



if __name__ == "__main__":
    
    img_dir = r"C:\Users\joortif\Desktop\datasets\Preprocesados\forest_orig_reduc\train\images"
    labels_dir = r"C:\Users\joortif\Desktop\datasets\Preprocesados\forest_orig_reduc\train\labels"
    output_path = r"C:\Users\joortif\Desktop\Resultados_ImageDatasetAnalyzer\sidewalk_train_resultados\huggingface"
    analysis_path = r"C:\Users\joortif\Desktop\Resultados_ImageDatasetAnalyzer\freiburg_resultados\analysis"

    os.makedirs(os.path.join(output_path, "kmeans"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "agglomerative"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "dbscan"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "optics"), exist_ok=True)

    kmeans_output_path = os.path.join(output_path, "kmeans")
    agglomerative_output_path = os.path.join(output_path, "agglomerative")
    dbscan_output_path = os.path.join(output_path, "dbscan")
    optics_output_path = os.path.join(output_path, "optics")

    os.makedirs(os.path.join(kmeans_output_path, "elbow"), exist_ok=True)
    os.makedirs(os.path.join(kmeans_output_path, "calinski"), exist_ok=True)

    kmeans_elbow_path = os.path.join(kmeans_output_path, "elbow")
    kmeans_silhouette_path = os.path.join(kmeans_output_path, "calinski")

    random_state = 123

    dataset = ImageDataset(img_dir)
    label_dataset = ImageLabelDataset(img_dir=img_dir, label_dir=labels_dir, background=0)

    #emb = HuggingFaceEmbedding("openai/clip-vit-base-patch16")
    #emb = PyTorchEmbedding("densenet121")
    #emb = OpenCVLBPEmbedding(8, 24, resize_height=224, resize_width=224)
    emb = TensorflowEmbedding("MobileNetV2")

    embeddings = emb.generate_embeddings(dataset)
    
    kmeans = KMeansClustering(dataset, embeddings, random_state)
    agglomerative = AgglomerativeClustering(dataset, embeddings, random_state)
    dbscan = DBSCANClustering(dataset, embeddings, random_state)
    optics = OPTICSClustering(dataset, embeddings, None)

    reduced_dataset_kmeans = kmeans.select_balanced_images(4, 0.7, diverse_percentage=0.5)
    reduced_dataset_agg = agglomerative.select_balanced_images(2, 'ward', 0.7, diverse_percentage=0.5)
    reduced_dataset_dbscan = dbscan.select_balanced_images(6.551724137931035, 18, 0.7, diverse_percentage=0.5)
    reduced_dataset_optics = optics.select_balanced_images(12, 0.7, diverse_percentage=0.5)
