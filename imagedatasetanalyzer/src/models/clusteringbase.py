import matplotlib.pyplot as plt
import numpy as np
import os
import random

from scipy.spatial.distance import cdist

import logging

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

import shutil

from imagedatasetanalyzer.src.datasets.imagedataset import ImageDataset

class ClusteringBase():

    """
    ClusteringBase class for performing clustering tasks on image datasets and generating visualizations.

    This class facilitates the process of performing clustering on a dataset of image embeddings. It offers methods for dimensionality reduction, 
    cluster visualization, selecting images from clusters, and evaluating clustering quality.

    Attributes:
        dataset (ImageDataset): Dataset of images used for clustering.
        embeddings (np.ndarray): Feature embeddings for each image.
        random_state (int): Random seed used for reproducibility.
    """

    def __init__(self, dataset: ImageDataset, embeddings: np.ndarray,random_state: int):
        """
        Args:
            dataset (ImageDataset): Dataset of images used for clustering.
            embeddings (np.ndarray): Feature embeddings for each image.
            random_state (int): Random seed used for reproducibility.
        """
        self.dataset = dataset
        self.embeddings = embeddings
        self.random_state = random_state

        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.hasHandlers():  
            handler = logging.StreamHandler()  
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def _evaluate_metric(self, metric: str):
        if metric == 'silhouette':
            scoring_function = silhouette_score
        elif metric == 'calinski':
            scoring_function = calinski_harabasz_score
        elif metric == 'davies':
            scoring_function = davies_bouldin_score
        else:
            raise ValueError(f"Unsupported metric: {metric}. Choose 'silhouette', 'calinski', or 'davies'.")
        
        return scoring_function

    def reduce_dimensions(self, method: str = 'tsne') -> np.ndarray:
        """
        Reduces the dimensionality of the embeddings for visualization purposes.
        """
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=self.random_state)
        elif method == 'pca':
            reducer = PCA(n_components=2, random_state=self.random_state)
        else:
            raise ValueError(f"Unsupported reduction method: {method}")
        
        return reducer.fit_transform(self.embeddings)

    def plot_clusters(self, embeddings_2d: np.ndarray, labels: np.ndarray, num_clusters: int, reduction: str, output: str = None):
        """
        Plots the clustering result in 2D after dimensionality reduction.
        """

        plt.figure(figsize=(12, 8))
        colormap = plt.cm.get_cmap('tab20')

        outliers = np.where(labels == -1)[0] 
        if outliers.size > 0:  
            num_clusters -= 1
            plt.scatter(
                embeddings_2d[outliers, 0], embeddings_2d[outliers, 1],
                label='Noise', color='black', marker='x', s=50
            )
        
        for i in range(num_clusters):
            if i != -1:  
                cluster_indices = np.where(labels == i)
                plt.scatter(embeddings_2d[cluster_indices, 0], embeddings_2d[cluster_indices, 1], 
                            label=f'Cluster {i}', color=colormap(i))

        plt.title(f'Cluster visualization with {reduction.upper()}')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.legend()
        plt.grid(True)

        if output:
            output = os.path.join(output, f"clustering_{reduction}.png")
            plt.savefig(output, bbox_inches='tight')
            print(f"Plot saved to {output}")
            plt.close()
        else:
            plt.show()

    def show_cluster_images(self, cluster_id: int, labels: np.ndarray, images_to_show: int=9, images_per_row: int=3, output: str=None) -> None:
        """
        Displays a grid of the first images_to_show number of images that belong to a specific cluster.

        Parameters:
            cluster_id (int): The identifier of the cluster whose images will be displayed.
            labels (np.ndarray): An array containing the cluster labels assigned to the images.
            dataset (ImageDataset): An object containing the images to display.
            images_to_show (int, optional): The maximum number of images to display. The default value is 9.
            images_per_row (int, optional): The number of images per row. The default value is 3.
            output (str, optional): The path to save the plot. If None, the plot will be displayed.

        Returns:
            None: This method does not return any value. It generates and either shows or saves a figure with the images.
        """

        if images_to_show <=0:
            return

        cluster_indices = [i for i, c in enumerate(labels) if c == cluster_id]

        num_images = len(cluster_indices)
        
        if images_to_show > 0:
            cluster_indices = cluster_indices[:images_to_show]

        if images_to_show > num_images:
            print(f"Parameter images_to_show={images_to_show} is greater than the number of images in cluster {cluster_id}. Showing {num_images} images.")
            images_to_show = len(cluster_indices)

        if num_images == 1:  
            image_idx = cluster_indices[0]
            image = self.dataset.get_image(image_idx)

            plt.figure(figsize=(6, 6))
            plt.imshow(image)
            plt.axis('off')
            plt.title(f'Cluster {cluster_id}: Single Image', fontsize=16)

            if output:
                output = os.path.join(output, f"cluster_{cluster_id}_images.png")
                plt.savefig(output, bbox_inches='tight')
                print(f"Plot saved to {output}")
                plt.close()
            else:
                plt.show()

            return

        num_rows = (images_to_show // images_per_row) + (1 if images_to_show % images_per_row != 0 else 0)

        plt.figure(figsize=(15, 5 * num_rows))
        plt.suptitle(f'Cluster {cluster_id}: {num_images} images', fontsize=16, y=1.02)

        for idx, image_idx in enumerate(cluster_indices):  
            plt.subplot(num_rows, images_per_row, idx + 1)

            image = self.dataset.get_image(image_idx)

            plt.imshow(image)
            plt.axis('off')
            plt.title(f'Image {idx}', fontsize=10)
        
        plt.tight_layout()

        if output:
            output = os.path.join(output, f"cluster_{cluster_id}_images.png")
            plt.savefig(output, bbox_inches='tight')
            print(f"Plot saved to {output}")
            plt.close()
        else:
            plt.show()


    def _calculate_medoids(self, embeddings=np.ndarray, labels=np.ndarray):
        """
        Calculates medoids for each cluster.
        """
        medoids = []
        for cluster_idx in np.unique(labels):
            
            if cluster_idx == -1: 
                continue
            
            cluster_embeddings = embeddings[labels == cluster_idx]
            
            if len(cluster_embeddings) == 0: 
                medoids.append(None)
                continue
            
            distances = cdist(cluster_embeddings, cluster_embeddings, metric="euclidean")
            medoid_idx = np.argmin(distances.sum(axis=1))
            medoids.append(cluster_embeddings[medoid_idx])
        
        return np.array([m for m in medoids if m is not None])
    
    def _select_images(self, cluster_embeddings, cluster_filenames, cluster_centers, cluster_idx,
                        num_selected_images, num_diverse_images, selection_type):
        """
        Selects representative or diverse images from a cluster.
        """

        if selection_type.lower() == "random":
            num_selected_images = min(num_selected_images, len(cluster_filenames))

            return random.sample(cluster_filenames.tolist(), num_selected_images)

        if cluster_centers is not None and len(cluster_centers) > cluster_idx and cluster_idx != -1:
            cluster_center = cluster_centers[cluster_idx]
        else:
            cluster_center = np.mean(cluster_embeddings, axis=0)  

        distances = np.linalg.norm(cluster_embeddings - cluster_center, axis=1)

        diverse_images = []

        if selection_type.lower() == "representative":
            closest_indices = distances.argsort()[:num_selected_images - num_diverse_images]
            selected_images = cluster_filenames[closest_indices]
        else:
            farthest_indices = distances.argsort()[-(num_selected_images - num_diverse_images):]
            selected_images = cluster_filenames[farthest_indices]
        
        if num_diverse_images > 0:
            farthest_indices = distances.argsort()[-num_diverse_images:]
            diverse_images = cluster_filenames[farthest_indices]

        return np.unique(np.concatenate((selected_images, diverse_images)))

    def _select_balanced_images(self, labels: np.ndarray, cluster_centers: np.ndarray | None, reduction: float, selection_type: str,
                               diverse_percentage: float, include_outliers: bool, output_directory: str):
        """
        Selects a balanced subset of images.

        Args:
            labels (np.ndarray): Cluster labels for each image, obtained from a clustering algorithm. Each label corresponds to the cluster 
                                an image belongs to. Outliers are marked with the label -1.
            cluster_centers (np.ndarray | None): Centroids or medoids of the clusters. If None, the cluster centers are calculated dynamically 
                                                based on the current clustering. Defaults to None.
            reduction (float): The fraction of images to retain. Must be a value between 0 and 1, where 0 means no images are retained, and 1 
                            means all images are kept.
            selection_type (str): Specifies the selection strategy. Options include:
                                - "representative": Selects images closest to the cluster center (most typical of the cluster).
                                - "diverse": Selects images farthest from the cluster center (maximizing diversity).
                                - "random": Selects images randomly from each cluster.
                                Defaults to "representative".
            diverse_percentage (float): Specifies the percentage of selected images to prioritize diversity when `selection_type` is "representative".
                                        Must be a value between 0 and 1.
            include_outliers (bool): Determines whether images marked as outliers (label -1) are included in the selection process. If True, 
                                    outliers are included in the reduced dataset; otherwise, they are excluded.
            output_directory (str): Path to the directory where the reduced dataset will be saved. The reduced dataset includes the selected 
                                    images.

        Returns:
            ImageDataset: Reduced dataset instance.
        """
        if selection_type.lower() not in ['representative', 'diverse', 'random']:
            raise ValueError("Invalid value for selection_type. Must be 'representative', 'diverse' or 'random'.")

        if cluster_centers is None and selection_type.lower() != 'random':
            self._calculate_medoids(self.embeddings, labels)

        if selection_type.lower() != 'representative':
            diverse_percentage = 0

        total_images = len(self.dataset)
        images_embeddings = self.embeddings.copy()

        if -1 in labels and not include_outliers:
            print("Skipping outliers...")
            
            valid_indices = labels != -1
            labels = labels[valid_indices]  
            images_embeddings = self.embeddings[valid_indices] 

            total_images = len(images_embeddings)
        
        num_selected_images_total = int(total_images * reduction)
        group_sizes = [np.sum(labels == cluster_idx) for cluster_idx in np.unique(labels)]

        if output_directory:
            os.makedirs(output_directory, exist_ok=True)

        reduced_dataset_files = []

        for cluster_idx in np.unique(labels):
            
            cluster_embeddings = images_embeddings[labels == cluster_idx]
            cluster_filenames = np.array([self.dataset.image_files[i] for i in range(total_images) if labels[i] == cluster_idx])

            proportion = group_sizes[cluster_idx] / total_images
            num_selected_images = int(proportion * num_selected_images_total)
            num_diverse_images = int(num_selected_images * diverse_percentage)
            num_selected_images = min(num_selected_images, len(cluster_filenames))

            selected_images = self._select_images(cluster_embeddings, cluster_filenames, cluster_centers, cluster_idx,
                                                   num_selected_images, num_diverse_images, selection_type)

            reduced_dataset_files.extend(selected_images)

            if output_directory:
                for filename in selected_images:
                    src_path = os.path.join(self.dataset.img_dir, filename)
                    dst_path = os.path.join(output_directory, filename)
                    shutil.copy(src_path, dst_path)

            print(f"Cluster {cluster_idx}: {len(selected_images)} images selected from {len(cluster_embeddings)}.")
            if diverse_percentage > 0:
                print(f"  - {num_diverse_images} diverse images and {num_selected_images - num_diverse_images} representative images.")
            else:
                print(f" - {num_selected_images} {selection_type} images.")

        if output_directory: 
            print(f"Reduced dataset saved to: {output_directory}")
        else:
            output_directory = self.dataset.img_dir

        reduced_dataset = ImageDataset(output_directory, reduced_dataset_files)

        print(f"Dataset reduced to {len(reduced_dataset)} images.")

        return reduced_dataset

