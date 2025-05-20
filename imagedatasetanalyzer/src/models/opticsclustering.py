import os
from sklearn.cluster import OPTICS
import numpy as np

import matplotlib.pyplot as plt

from imagedatasetanalyzer.src.models.clusteringbase import ClusteringBase
from imagedatasetanalyzer.src.datasets.imagedataset import ImageDataset


class OPTICSClustering(ClusteringBase):
    """
    Performs data clustering using the OPTICS algorithm. This class allows evaluating clustering performance based on different metrics, 
    applying OPTICS clustering to the data, and selecting a balanced subset of images from the dataset based on the clustering results.

    Attributes:
        dataset (ImageDataset): Dataset of images used for clustering.
        embeddings (np.ndarray): Feature embeddings for each image.
        random_state (int): Random seed used for reproducibility.
    """

    def find_best_OPTICS(self,min_samples_range: range, metric: str='silhouette', plot: bool=True, output: str=None, verbose: bool=False):
        """
        Evaluates OPTICS clustering using the specified metric, including noise points.

        Parameters:
            min_samples_range (range): The range of 'min_samples' values to evaluate.
            metric (str, optional): The evaluation metric to use ('silhouette', 'calinski', 'davies'). Defaults to 'silhouette'.
            plot (bool, optional): Whether to plot the results. Defaults to True.
            output (str, optional): Path to save the plot as an image. If None, the plot is displayed.
            verbose (bool, optional): Whether to display warnings and status messages in the logs. If True, logs are written using the logger. Defaults to False.

        Returns:
            tuple: The best 'min_samples' and the best score.
        """

        scoring_function = self._evaluate_metric(metric)
        results = []
        
        for min_samples in min_samples_range:
            optics = OPTICS(min_samples=min_samples)
            labels = optics.fit_predict(self.embeddings)
                
            if np.all(labels == -1):
                if verbose:
                    self.logger.warning("No clusters found for min_samples=%s. All points are noise.", min_samples)
                results.append((min_samples, float('inf') if metric == 'davies' else 0, labels))
                continue

            unique_labels = np.unique(labels)
            if len(unique_labels) == len(self.embeddings):
                if verbose:
                    self.logger.warning("Each point is assigned to its own cluster for min_samples=%s.", min_samples)
                results.append((min_samples, float('inf') if metric == 'davies' else 0, labels))
                continue

            valid_indices = labels != -1
            valid_labels = labels[valid_indices]
            valid_embeddings = self.embeddings[valid_indices]

            if len(np.unique(valid_labels)) == 1:
                if verbose:
                    self.logger.warning("Only one cluster and noise cluster found for min_samples=%s. Can't compute %s score.", min_samples, metric.lower())
                results.append((min_samples, float('inf') if metric == 'davies' else 0, labels))
                continue

            score = scoring_function(valid_embeddings, valid_labels)
            results.append((min_samples, score, labels))
        
        scores = [score for _, score, _ in results]

        if all(score == 0 for score in scores) or all(score == float('inf') for score in scores):
            if verbose:
                self.logger.warning("No valid clustering found for the ranges given. Try adjusting the parameters for better clustering.")
            plot = False

        if plot:
            plt.figure(figsize=(10, 7))
            plt.plot(min_samples_range, scores, marker='o', linestyle='--')
            plt.title(f'OPTICS evaluation ({metric.capitalize()} Score)')
            plt.xlabel('Min samples')
            plt.ylabel(f'{metric.capitalize()} Score')
            plt.xticks(min_samples_range)
            plt.grid(True)
            
            if output:
                output = os.path.join(output, f"optics_evaluation_{metric.lower()}.png")
                plt.savefig(output, format='png')
                print(f"Plot saved to {output}")
                plt.close()


        best_min_samples, best_score, labels = max(results, key=lambda x: x[1]) if metric != 'davies' else min(results, key=lambda x: x[1])

        return best_min_samples, best_score, labels

    def clustering(self, min_samples: int = 5, reduction: str = 'tsne', output: str = None) -> np.ndarray:
        """
        Apply OPTICS clustering to the embeddings.
        
        Parameters:
            min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
            reduction (str): Dimensionality reduction method ('tsne' or 'pca'). Defaults to 'tsne'.
            output (str): Path to save the plot as an image. If None, the plot is displayed.
        
        Returns:
            np.ndarray: Cluster labels assigned to each data point.
        """
        optics = OPTICS(min_samples=min_samples)
        labels = optics.fit_predict(self.embeddings)

        embeddings_2d = self.reduce_dimensions(reduction)

        num_clusters = len(set(labels))

        self.plot_clusters(embeddings_2d, labels, num_clusters, reduction, 'optics', output)

        return labels
    
    def select_balanced_images(self, min_samples: int=5, reduction: float=0.5, selection_type: str = "representative", 
                               diverse_percentage: float = 0.1, include_outliers: bool=False, existing_labels: np.ndarray = None, output_directory: str = None) -> ImageDataset:
        """
        Selects a subset of images from a dataset based on OPTICS clustering.
        The selection can be either representative (closest to centroids) or diverse (farthest from centroids).

        Args:
            min_samples (float): The minimum number of samples required to form a cluster in OPTICS.
            reduction (float, optional): Percentage of the total dataset to retain. Defaults to 0.5. A value of 0.5 retains 50% of the dataset.  
            selection_type (str, optional): Determines whether to select "representative" or "diverse" images. Defaults to "representative".
            diverse_percentage (float, optional): Percentage of the cluster's images to select as diverse. Defaults to 0.1.
            include_outliers (bool): Whether to include outliers (label -1) in the selection. Defaults to False.
            output_directory (str, optional): Directory to save the reduced dataset. If None, the folder will not be created.

        Returns:
            ImageDataset: A new `ImageDataset` instance containing the reduced set of images.
        """
        if existing_labels is None:
            optics = OPTICS(min_samples=min_samples)
            existing_labels = optics.fit_predict(self.embeddings)

        reduced_dataset_optics = self._select_balanced_images(labels=existing_labels, cluster_centers=None, reduction=reduction, selection_type=selection_type, diverse_percentage=diverse_percentage, 
                                                              include_outliers=include_outliers, output_directory=output_directory)

        return reduced_dataset_optics