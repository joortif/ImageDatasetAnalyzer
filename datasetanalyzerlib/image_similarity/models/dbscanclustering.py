from sklearn.cluster import DBSCAN

from datasetanalyzerlib.image_similarity.models.clusteringbase import ClusteringBase

import matplotlib.pyplot as plt

import os

import numpy as np

class DBSCANClustering(ClusteringBase):

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
    
    def find_best_DBSCAN(self, eps_range: range, min_samples_range: range, metric: str='silhouette', plot: bool=True, output: str=None) -> tuple:
        """
        Evaluates DBSCAN clustering using the specified metric, including noise points.

        Parameters:
            eps_range (range): The range of 'eps' values to evaluate.
            min_samples_range (range): The range of 'min_samples' values to evaluate.
            metric (str, optional): The evaluation metric to use ('silhouette', 'calinski', 'davies'). Defaults to 'silhouette'.
            plot (bool, optional): Whether to plot the results. Defaults to True.
            output (str, optional): Path to save the plot as an image. If None, the plot is displayed.

        Returns:
            tuple: The best 'eps', the best 'min_samples', and the best score.
        """

        scoring_function = self.evaluate_metric(metric)
        results = []
        scores_by_eps = {eps: [] for eps in eps_range}
        scores_by_min_samples = {min_samples: [] for min_samples in min_samples_range}

        for eps in eps_range:
            for min_samples in min_samples_range:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(self.embeddings)
                
                if np.all(labels == -1):
                    print(f"Warning: No clusters found for eps={eps}, min_samples={min_samples}. All points are considered noise.")
                    results.append((eps, min_samples, -1))  
                    continue

                unique_labels = np.unique(labels)
                if len(unique_labels) == len(self.embeddings):
                    print(f"Warning: Each point is assigned to its own cluster for eps={eps}, min_samples={min_samples}. This may not be meaningful.")
                    results.append((eps, min_samples, -1))  
                    continue

                valid_indices = labels != -1  
                valid_labels = labels[valid_indices]
                valid_embeddings = self.embeddings[valid_indices]

                if len(np.unique(valid_labels)) > 1:  
                    score = scoring_function(valid_embeddings, valid_labels)

                    results.append((eps, min_samples, score))
                    scores_by_eps[eps].append(score)
                    scores_by_min_samples[min_samples].append(score)
        
        best_combination = max(results, key=lambda x: x[2]) if metric != 'davies' else min(results, key=lambda x: x[2])
        best_eps, best_min_samples, best_score = best_combination

        if plot:
            plt.figure(figsize=(10, 6))

            valid_eps = [eps for eps, score in zip(eps_range, scores_by_min_samples[min_samples]) if score is not None]
            valid_scores = [score for score in scores_by_min_samples[min_samples] if score is not None]

            if valid_eps:  
                plt.plot(valid_eps, valid_scores, marker='o', linestyle='--', label=f'Min Samples: {min_samples}')

            plt.title(f'DBSCAN Clustering Evaluation ({metric.capitalize()} Score)')
            plt.xlabel('Epsilon (eps)')
            plt.ylabel(f'{metric.capitalize()} Score')
            plt.grid(True)
            plt.legend()

            if output:
                output = os.path.join(output, f"dbscan_clustering_evaluation_{metric.lower()}.png")
                plt.savefig(output, format='png')
                print(f"Plot saved to {output}")
            else:
                plt.show()

        return best_eps, best_min_samples, best_score
