import os

import sklearn.cluster

import matplotlib.pyplot as plt
import numpy as np

from datasetanalyzerlib.image_similarity.models.clusteringbase import ClusteringBase

class AgglomerativeClustering(ClusteringBase):

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

                scoring_function = self.evaluate_metric(metric)

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
            else:
                plt.show()

        return best_k, best_linkage, best_score


    def clustering(self, k: int, linkage: str, reduction='tsne', output: str=None) -> np.ndarray:
        """
        Applies AgglomerativeClustering clustering to the given embeddings, reduces dimensionality for visualization, 
        and optionally saves or displays a scatter plot of the clusters.

        Parameters:
            k (int): Number of clusters for AgglomerativeClustering: 'ward', 'complete', 'average' or 'single'. Defaults to 'ward'. 
            linkage (str): Type of linkage to use with AgglomerativeClustering.
            reduction (str, optional): Dimensionality reduction method ('tsne' or 'pca'). Defaults to 'tsne'.
            output (str, optional): Path to save the plot as an image. If None, the plot is displayed.
            
        Returns:
            array: Cluster labels assigned by KMeans for each data point.
        """
        aggClusteringModel = sklearn.cluster.AgglomerativeClustering(n_clusters=k, linkage=linkage)
        labels = aggClusteringModel.fit_predict(self.embeddings)

        embeddings_2d = self.reduce_dimensions(reduction)

        self.plot_clusters(embeddings_2d, labels, k, reduction, output)

        return labels