import os

from kneed import KneeLocator
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import numpy as np

from imagedatasetanalyzer.src.models.clusteringbase import ClusteringBase
from imagedatasetanalyzer.src.datasets.imagedataset import ImageDataset

class KMeansClustering(ClusteringBase): 
    """
    Performs data clustering using the KMeans algorithm. This class allows finding the optimal number of clusters 
    using the elbow rule, evaluating the clustering quality using various metrics, applying clustering to the data, 
    and selecting a balanced subset of images based on the clustering results.

    Attributes:
        dataset (ImageDataset): Dataset of images used for clustering.
        embeddings (np.ndarray): Feature embeddings for each image.
        random_state (int): Random seed used for reproducibility.
    """

    def find_elbow(self, clusters_max: int, plot: bool=True, output: str=None) -> int:
        """
        Applies the elbow rule to determine the optimal number of clusters for KMeans clustering.
        
        Parameters:
            embeddings (array): The data to be clustered, typically a 2D array of embeddings.
            clusters_max (int): The maximum number of clusters to evaluate (k).
            random_state (int): The random seed for reproducibility of KMeans results.
            plot (bool, optional): Whether to generate and display/save the elbow plot. Defaults to True.
            output (str, optional): Path to save the generated plot as an image. 
                                    If None, the plot will not be saved.
            
        Returns:
            int: The optimal number of clusters determined by the elbow rule.
        """
        inertia_values = []
        interval = range(2, clusters_max)

        for k in interval:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state)
            kmeans.fit(self.embeddings)
            inertia_values.append(kmeans.inertia_)

        if plot:
            plt.figure(figsize=(10, 7))
            plt.plot(interval, inertia_values, marker='o', linestyle='--')
            plt.title('Elbow rule for KMeans')
            plt.xlabel('Num clusters (k)')
            plt.ylabel('Inertia')
            plt.xticks(interval)
            plt.grid(True)
            
            if output:
                output = os.path.join(output, "kmeans_elbow.png")
                plt.savefig(output, format='png')
                print(f"Plot saved to {output}")
                plt.close()

        
        knee_locator = KneeLocator(interval, inertia_values, curve="convex", direction="decreasing")
        best_k = knee_locator.knee

        return best_k

    def find_best_n_clusters(self, n_clusters_range: range, metric: str, plot: bool=True, output: str=None) -> tuple:
        """
        Evaluates KMeans clustering using the specified metric.

        Parameters:
            n_clusters_range (range): The range of 'n_clusters' values to evaluate.
            metric (str, optional): The evaluation metric to use ('silhouette', 'calinski', 'davies'). Defaults to 'silhouette'.
            plot (bool, optional): Whether to plot the results. Defaults to True.
            output (str, optional): Path to save the plot as an image. If None, the plot is displayed.

        Returns:
            tuple: The best k and the best score.
        """

        scoring_function = self._evaluate_metric(metric)
        results = []
        
        for k in n_clusters_range:
            kmeans = KMeans(n_clusters=k)
            labels = kmeans.fit_predict(self.embeddings)

            score = scoring_function(self.embeddings, labels)
            results.append((k, score, labels))
        
        scores = [score for _, score, _ in results]

        if plot:
            plt.figure(figsize=(10, 7))
            plt.plot(n_clusters_range, scores, marker='o', linestyle='--')
            plt.title(f'KMeans evaluation ({metric.capitalize()} Score)')
            plt.xlabel('n_clusters values')
            plt.ylabel(f'{metric.capitalize()} Score')
            plt.xticks(n_clusters_range)
            plt.grid(True)
            
            if output:
                output = os.path.join(output, f"kmeans_evaluation_{metric.lower()}.png")
                plt.savefig(output, format='png')
                print(f"Plot saved to {output}")
                plt.close()

            

        best_n_clusters, best_score, best_labels = max(results, key=lambda x: x[1]) if metric != 'davies' else min(results, key=lambda x: x[1])

        return best_n_clusters, self.random_state, best_score, best_labels

    def clustering(self, n_clusters: int, reduction='tsne',  output: str=None) -> np.ndarray:
        """
        Applies KMeans clustering to the given embeddings, reduces dimensionality for visualization, 
        and optionally saves or displays a scatter plot of the clusters.

        Parameters:
            embeddings (array): High-dimensional data to be clustered, typically a 2D array.
            n_clusters (int): Number of clusters for KMeans.
            output (str, optional): Path to save the plot as an image. If None, the plot is displayed.
            reduction (str, optional): Dimensionality reduction method ('tsne' or 'pca'). Defaults to 'tsne'.

        Returns:
            array: Cluster labels assigned by KMeans for each data point.
        """
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state)
        labels = kmeans.fit_predict(self.embeddings)

        embeddings_2d = self.reduce_dimensions(reduction)

        self.plot_clusters(embeddings_2d, labels, n_clusters, reduction, 'kmeans', output)
        
        return labels
    
    def select_balanced_images(self, n_clusters: int=3, reduction: float=0.5, selection_type: str = "representative", 
                               diverse_percentage: float = 0.1, output_directory: str = None) -> ImageDataset:
        """
        Selects a subset of images from a dataset based on KMeans clustering and its centroids.
        The selection can be either representative (closest to centroids) or diverse (farthest from centroids).

        Args:
            n_clusters (int): Number of clusters for KMeans.
            reduction (float, optional): Proportion of the dataset to retain. Defaults to 0.5. A value of 0.5 retains 50% of the dataset. 
            selection_type (str, optional): Determines whether to select "representative" or "diverse" images. Defaults to "representative".
            diverse_percentage (float, optional): Percentage of images selected as diverse within each cluster. Defaults to 0.1.
            output_directory (str, optional): Directory to save the reduced dataset. If None, the folder will not be created.

        Returns:
            ImageDataset: A new `ImageDataset` instance containing the reduced set of images.
        """
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state)
        labels = kmeans.fit_predict(self.embeddings)
        cluster_centers = kmeans.cluster_centers_

        reduced_dataset_kmeans = self._select_balanced_images(labels, cluster_centers, reduction, selection_type, diverse_percentage, False, output_directory)

        return reduced_dataset_kmeans