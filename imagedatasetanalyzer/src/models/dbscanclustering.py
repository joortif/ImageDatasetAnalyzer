from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import os

from imagedatasetanalyzer.src.models.clusteringbase import ClusteringBase
from imagedatasetanalyzer.src.datasets.imagedataset import ImageDataset

import numpy as np

class DBSCANClustering(ClusteringBase):
    """
    DBSCANClustering class for performing DBSCAN clustering tasks on image datasets.

    This class is designed to apply DBSCAN (Density-Based Spatial Clustering of Applications with Noise) on image embeddings, 
    evaluate clustering configurations, visualize the clustering results, and select representative or diverse subsets 
    of images based on the clustering results.

    Attributes:
        dataset (ImageDataset): Dataset of images used for clustering.
        embeddings (np.ndarray): Feature embeddings for each image.
        random_state (int): Random seed used for reproducibility.
    """
    
    def find_best_DBSCAN(self, eps_range: range, min_samples_range: range, metric: str='silhouette', plot: bool=True, output: str=None, verbose: bool=False) -> tuple:
        """
        Evaluates DBSCAN clustering using the specified metric, including noise points.

        Parameters:
            eps_range (range): The range of 'eps' values to evaluate.
            min_samples_range (range): The range of 'min_samples' values to evaluate.
            metric (str, optional): The evaluation metric to use ('silhouette', 'calinski', 'davies'). Defaults to 'silhouette'.
            plot (bool, optional): Whether to plot the results. Defaults to True.
            output (str, optional): Path to save the plot as an image. If None, the plot is displayed.
            verbose (bool, optional): Whether to display warnings and status messages in the logs. If True, logs are written using the logger. Defaults to False.

        Returns:
            tuple: The best 'eps', the best 'min_samples', and the best score.
        """

        scoring_function = self._evaluate_metric(metric)
        results = []
        for eps in eps_range:
            for min_samples in min_samples_range:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(self.embeddings)
                
                if np.all(labels == -1):
                    if verbose:
                        self.logger.warning(f"No clusters found for eps={eps}, min_samples={min_samples}. All points are noise.")
                    results.append((eps, min_samples, float('inf') if metric == 'davies' else -1))
                    continue

                unique_labels = np.unique(labels)
                if len(unique_labels) == len(self.embeddings):
                    if verbose:
                        self.logger.warning(f"Each point is assigned to its own cluster for eps={eps}, min_samples={min_samples}.")
                    results.append((eps, min_samples, float('inf') if metric == 'davies' else -1))
                    continue

                valid_indices = labels != -1
                valid_labels = labels[valid_indices]
                valid_embeddings = self.embeddings[valid_indices]

                if len(np.unique(valid_labels)) == 1:
                    if verbose:
                        self.logger.warning(f"Only 1 cluster found for eps={eps}, min_samples={min_samples}. Can't calculate metric {metric.lower()}.")
                    results.append((eps, min_samples, float('inf') if metric == 'davies' else -1))
                    continue

                score = scoring_function(valid_embeddings, valid_labels)
                results.append((eps, min_samples, score))

        best_combination = max(results, key=lambda x: x[2]) if metric != 'davies' else min(results, key=lambda x: x[2])
        best_eps, best_min_samples, best_score = best_combination

        if best_score == -1:
            if verbose:
                self.logger.warning(f"No valid clustering found for the ranges given. Try adjusting the parameters for better clustering.")
            return best_eps, best_min_samples, best_score

        filtered_min_samples = list(min_samples_range)[:9]
        num_plots = len(filtered_min_samples)

        if plot:
            if num_plots > 0:
                ncols = min(num_plots, 3)
                nrows = (num_plots + ncols - 1) // ncols

                fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), sharey=True)
                axes = axes.flatten()

                for i, ax in enumerate(axes[:num_plots]):
                    min_samples = filtered_min_samples[i]
                    scores_for_min_samples = [(eps, score) for eps, ms, score in results if ms == min_samples]

                    if scores_for_min_samples:
                        eps_values, scores = zip(*scores_for_min_samples)

                        ax.plot(eps_values, scores, marker='o', label=f'min_samples={min_samples}')
                        ax.set_title(f'min_samples={min_samples}')
                        ax.set_xlabel('Eps')
                        ax.set_ylabel(f'{metric.capitalize()} Score')
                        ax.grid(True)
                        ax.legend()

                for j in range(num_plots, len(axes)):
                    axes[j].axis('off')

                if output:
                    output = os.path.join(output, f"dbscan_evaluation_{metric.lower()}.png")
                    plt.savefig(output, format='png')
                    print(f"Plot saved to {output}")
                    plt.close()
                else:
                    plt.show()

            return best_eps, best_min_samples, best_score
    
    def clustering(self, eps: float = 0.5, min_samples: int = 5, reduction: str = 'tsne', output: str = None) -> np.ndarray:
        """
        Apply DBSCAN clustering to the embeddings.
        
        Parameters:
            eps (float): The maximum distance between two samples for them to be considered as in the same neighborhood.
            min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
            reduction (str): Dimensionality reduction method ('tsne' or 'pca'). Defaults to 'tsne'.
            output (str): Path to save the plot as an image. If None, the plot is displayed.
        
        Returns:
            np.ndarray: Cluster labels assigned to each data point.
        """
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(self.embeddings)

        embeddings_2d = self.reduce_dimensions(reduction)

        num_clusters = len(set(labels))

        self.plot_clusters(embeddings_2d, labels, num_clusters, reduction, output)

        return labels
    
    def select_balanced_images(self, eps: float, min_samples: int, reduction: float=0.5, selection_type: str = "representative", 
                               diverse_percentage: float = 0.1, include_outliers: bool=False, output_directory: str = None) -> ImageDataset:
        """
        Selects a subset of images from a dataset based on DBSCAN clustering.
        The selection can be either representative (closest to centroids) or diverse (farthest from centroids).

        Args:
            eps (float): The maximum distance between two samples for them to be considered as in the same neighborhood.
            min_samples (float): The minimum number of samples required to form a cluster in DBSCAN.
            reduction (float, optional): Percentage of the total dataset to retain. Defaults to 0.5. A value of 0.5 retains 50% of the dataset.  
            selection_type (str, optional): Determines whether to select "representative" or "diverse" images. Defaults to "representative".
            diverse_percentage (float, optional): Percentage of the cluster's images to select as diverse.  Defaults to 0.1.
            include_outliers (bool): Whether to include outliers (label -1) in the selection. Defaults to False.
            output_directory (str, optional): Directory to save the reduced dataset. If None, the folder will not be created.

        Returns:
            ImageDataset: A new `ImageDataset` instance containing the reduced set of images.
        """
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(self.embeddings)

        reduced_dataset_dbscan = self._select_balanced_images(labels, None, reduction=reduction, selection_type=selection_type, diverse_percentage=diverse_percentage, 
                                                              include_outliers=include_outliers, output_directory=output_directory)

        return reduced_dataset_dbscan