import os

import sklearn.cluster

import matplotlib.pyplot as plt
import numpy as np

from imagedatasetanalyzer.src.models.clusteringbase import ClusteringBase
from imagedatasetanalyzer.src.datasets.imagedataset import ImageDataset

class AgglomerativeClustering(ClusteringBase):
    """
    AgglomerativeClustering class for performing agglomerative clustering tasks on image datasets.

    This class is designed to perform agglomerative clustering on image embeddings, evaluate different cluster configurations, 
    and provide methods for visualizing the clustering results, selecting representative or diverse subsets of images, 
    and evaluating clustering quality using various metrics.

    Attributes:
        dataset (ImageDataset): Dataset of images used for clustering.
        embeddings (np.ndarray): Feature embeddings for each image.
        random_state (int): Random seed used for reproducibility.
    """

    def find_best_agglomerative_clustering(self, n_clusters_range: range, metric: str='silhouette', linkages=None, plot=True, output: str=None) -> tuple: 
        """
        Evaluates Agglomerative Clustering using the specified metric.

        Parameters:
            n_clusters_range (range): The range of cluster numbers to evaluate.
            linkages (list, optional): The linkage criteria to evaluate. Defaults to ['ward', 'complete', 'average', 'single'].
            metric (str, optional): The evaluation metric to use ('silhouette', 'calinski', 'davies').
                                    Defaults to 'silhouette'.
            plot (bool, optional): Whether to plot the results. Defaults to True.

        Returns:
            tuple: The best number of clusters, the best linkage method, and the best score.
        """

        if not linkages:
            linkages = ['ward','complete','average','single']

        results = []
        scores_by_linkage = {linkage: [] for linkage in linkages}

        for linkage in linkages:
            for k in n_clusters_range:
                agglomerative = sklearn.cluster.AgglomerativeClustering(n_clusters=k, linkage=linkage)
                agglomerative_labels = agglomerative.fit_predict(self.embeddings)

                scoring_function = self._evaluate_metric(metric)

                score = scoring_function(self.embeddings, agglomerative_labels)
                scores_by_linkage[linkage].append(score)

                results.append((k, linkage, score))

        best_k, best_linkage, best_score = max(results, key=lambda x: x[2]) if metric != 'davies' else min(results, key=lambda x: x[2])

        if plot:
            plt.figure(figsize=(10, 6))
            for linkage in linkages:
                plt.plot(n_clusters_range, scores_by_linkage[linkage], marker='o', linestyle='--', label=f'Linkage: {linkage}')
            plt.title(f'Agglomerative Clustering evaluation ({metric.capitalize()} Score)')
            plt.xlabel('Number of Clusters')
            plt.ylabel(f'{metric.capitalize()} Score')
            plt.grid(True)
            plt.legend()

            if output:
                output = os.path.join(output, f"agglomerative_clustering_evaluation_{metric.lower()}.png")
                plt.savefig(output, format='png')
                print(f"Plot saved to {output}")
                plt.close()
            else:
                plt.show()

        return best_k, best_linkage, best_score


    def clustering(self, num_clusters: int, linkage: str, reduction='tsne', output: str=None) -> np.ndarray:
        """
        Applies AgglomerativeClustering clustering to the given embeddings, reduces dimensionality for visualization, 
        and optionally saves or displays a scatter plot of the clusters.

        Parameters:
            num_clusters (int): Number of clusters for AgglomerativeClustering.  
            linkage (str): Type of linkage to use with AgglomerativeClustering: 'ward', 'complete', 'average' or 'single'.
            reduction (str, optional): Dimensionality reduction method ('tsne' or 'pca'). Defaults to 'tsne'.
            output (str, optional): Path to save the plot as an image. If None, the plot is displayed.
            
        Returns:
            array: Cluster labels assigned by KMeans for each data point.
        """
        aggClusteringModel = sklearn.cluster.AgglomerativeClustering(n_clusters=num_clusters, linkage=linkage)
        labels = aggClusteringModel.fit_predict(self.embeddings)

        embeddings_2d = self.reduce_dimensions(reduction)

        self.plot_clusters(embeddings_2d, labels, num_clusters, reduction, output)

        return labels
    
    def select_balanced_images(self, n_clusters: int, linkage: str, reduction: float=0.5, selection_type: str = "representative", 
                               diverse_percentage: float = 0.1, output_directory: str = None) -> ImageDataset:
        """
        Selects a subset of images from a dataset based on AgglomerativeClustering.
        The selection can be either representative (closest to centroids) or diverse (farthest from centroids).

        Args:
            n_clusters (int): Number of clusters for AgglomerativeClustering.
            linkage (str): Type of linkage to use with AgglomerativeClustering.
            reduction (float, optional): Percentage of the total dataset to retain. Defaults to 0.5. A value of 0.5 retains 50% of the dataset. 
            selection_type (str, optional): Determines whether to select "representative" or "diverse" images. Defaults to "representative".
            diverse_percentage (float, optional): Percentage of the cluster's images to select as diverse. Defaults to 0.1.
            output_directory (str, optional): Directory to save the reduced dataset. If None, the folder will not be created.

        Returns:
            ImageDataset: A new `ImageDataset` instance containing the reduced set of images.
        """
        agglomerative = sklearn.cluster.AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        labels = agglomerative.fit_predict(self.embeddings)

        reduced_dataset_agglomerative = self._select_balanced_images(labels, None, reduction=reduction, selection_type=selection_type, diverse_percentage=diverse_percentage, 
                                                              include_outliers=False, output_directory=output_directory)

        return reduced_dataset_agglomerative