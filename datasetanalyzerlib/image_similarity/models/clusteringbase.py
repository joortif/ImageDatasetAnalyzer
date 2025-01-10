import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from datasetanalyzerlib.image_similarity.dataset import Dataset

class ClusteringBase():

    def __init__(self, embeddings: np.ndarray, random_state: int):
        self.embeddings = embeddings
        self.random_state = random_state

    def evaluate_metric(self, metric: str):
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

    def plot_clusters(self, embeddings_2d: np.ndarray, labels: np.ndarray, k: int, reduction: str, output: str = None):
        """
        Plots the clustering result in 2D after dimensionality reduction.
        """
        plt.figure(figsize=(12, 8))
        colormap = plt.cm.get_cmap('tab20')

        outliers = np.where(labels == -1)
        plt.scatter(embeddings_2d[outliers, 0], embeddings_2d[outliers, 1], 
                label='Noise', color='black', marker='x', s=50)
        if outliers:
            k -=1

        for i in range(k):
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
        else:
            plt.show()

    def show_cluster_images(cluster_id: int, labels: np.ndarray, dataset: Dataset, images_to_show: int=9, images_per_row: int=3, output: str=None) -> None:
        """
        Displays a grid of the first images_to_show number of images that belong to a specific cluster.

        Parameters:
            cluster_id (int): The identifier of the cluster whose images will be displayed.
            labels (np.ndarray): An array containing the cluster labels assigned to the images.
            dataset (Dataset): An object containing the images to display.
            images_to_show (int, optional): The maximum number of images to display. The default value is 9.
            images_per_row (int, optional): The number of images per row. The default value is 3.
            output (str, optional): The path to save the plot. If None, the plot will be displayed.

        Returns:
            None: This method does not return any value. It generates and either shows or saves a figure with the images.
        """

        if images_to_show <=0:
            return

        cluster_indices = [i for i, c in enumerate(labels) if c == id]

        num_images = len(cluster_indices)
        
        if images_to_show > 0:
            cluster_indices = cluster_indices[:images_to_show]

        if images_to_show > cluster_indices:
            print(f"Parameter images_to_show={images_to_show} is greater than the number of images in cluster {cluster_id}. Showing {len(cluster_indices)} images.")
            images_to_show = len(cluster_indices)

        num_rows = (len(cluster_indices) // images_per_row) + (1 if len(cluster_indices) % images_per_row != 0 else 0)

        plt.figure(figsize=(15, 5 * num_rows))
        plt.suptitle(f'Cluster {id}: {num_images} images', fontsize=16, y=1.02)

        for idx, image_idx in enumerate(cluster_indices):  
            plt.subplot(num_rows, images_per_row, idx + 1)

            image = dataset.get_image(image_idx)

            plt.imshow(image)
            plt.axis('off')
            plt.title(f'Image {idx}', fontsize=10)
        
        plt.tight_layout()

        if output:
            output = os.path.join(output, f"cluster_{cluster_id}_images.png")
            plt.savefig(output, bbox_inches='tight')
            print(f"Plot saved to {output}")
        else:
            plt.show()