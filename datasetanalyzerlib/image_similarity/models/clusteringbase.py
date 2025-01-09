import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

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