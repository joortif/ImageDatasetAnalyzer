from kneed import KneeLocator

import os

from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import numpy as np

from datasetanalyzerlib.image_similarity.models.clusteringbase import ClusteringBase

class KMeansClustering(ClusteringBase): 

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
            else:
                plt.show()
        
        knee_locator = KneeLocator(interval, inertia_values, curve="convex", direction="decreasing")
        best_k = knee_locator.knee

        return best_k

    def clustering(self, k: int, reduction='tsne',  output: str=None) -> np.ndarray:
        """
        Applies KMeans clustering to the given embeddings, reduces dimensionality for visualization, 
        and optionally saves or displays a scatter plot of the clusters.

        Parameters:
            embeddings (array): High-dimensional data to be clustered, typically a 2D array.
            k (int): Number of clusters for KMeans.
            random_state (int): Random seed for reproducibility.
            output (str, optional): Path to save the plot as an image. If None, the plot is displayed.
            reduction (str, optional): Dimensionality reduction method ('tsne' or 'pca'). Defaults to 'tsne'.

        Returns:
            array: Cluster labels assigned by KMeans for each data point.
        """
        kmeans = KMeans(n_clusters=k, random_state=self.random_state)
        labels = kmeans.fit_predict(self.embeddings)

        embeddings_2d = self.reduce_dimensions(reduction)

        self.plot_clusters(embeddings_2d, labels, k, reduction, output)
        
        return labels
    

