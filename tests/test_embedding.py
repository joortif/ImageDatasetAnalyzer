import sys
import os
import torch
import numpy as np


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'datasetanalyzerlib')))

from datasetanalyzerlib.image_similarity.embedding import Embedding
from datasetanalyzerlib.image_similarity.models.kmeansclustering import KMeansClustering
from datasetanalyzerlib.image_similarity.models.agglomerativeclustering import AgglomerativeClustering
from datasetanalyzerlib.image_similarity.models.dbscanclustering import DBSCANClustering
from datasetanalyzerlib.image_similarity.models.opticsclustering import OPTICSClustering


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb = Embedding("microsoft/swin-tiny-patch4-window7-224")
    dir = r"C:\Users\joortif\Desktop\datasets\forest_orig_reduc\train\images"
    output_path = r"C:\Users\joortif\Desktop\datasets\freiburg_resultados"

    kmeans_output_path = os.path.join(output_path, "kmeans")
    agglomerative_output_path = os.path.join(output_path, "agglomerative")
    dbscan_output_path = os.path.join(output_path, "dbscan")
    optics_output_path = os.path.join(output_path, "optics")


    random_state = 123

    #embeddings = emb.generate_embedding(dir)

    #np.save('embeddings.npy', embeddings)
    embeddings =  np.load('embeddings.npy')

    kmeans = KMeansClustering(embeddings, random_state)
    agglomerative = AgglomerativeClustering(embeddings, random_state)
    dbscan = DBSCANClustering(embeddings, random_state)
    optics = OPTICSClustering(embeddings, None)

    best_k = kmeans.find_elbow(15, output=kmeans_output_path)
    print("=============================================")
    print("KMeansClustering")
    print(f'Best K: {best_k}')
    labels_kmeans = kmeans.clustering(best_k, reduction='pca', output=kmeans_output_path)
    kmeans.clustering(best_k, reduction='tsne', output=kmeans_output_path)

    for cluster in np.unique(labels_kmeans):
        kmeans.show_cluster_images(cluster, labels_kmeans, )


    """"
    print("=============================================")
    
    best_k, best_linkage, best_score = agglomerative.find_best_agglomerative_clustering(
        n_clusters_range=range(2, 15), 
        metric='silhouette', 
        output=agglomerative_output_path
    )

    print("AgglomerativeClustering")
    print(f"Best K: {best_k}, Best Linkage: {best_linkage}, Best Score: {best_score}")

    labels_agg = agglomerative.clustering(
        k=best_k, 
        linkage=best_linkage, 
        reduction='pca', 
        output=agglomerative_output_path
    )

    agglomerative.clustering(
        k=best_k, 
        linkage=best_linkage, 
        reduction='tsne', 
        output=agglomerative_output_path
    )
    print("=============================================") """
    """ print("DBSCANClustering")
    best_eps, best_min_samples, best_score = dbscan.find_best_DBSCAN(
        eps_range= np.linspace(0.3, 2.0, 5),
        min_samples_range=np.arange(2, 10),
        metric='silhouette',
        plot=True,
        output=dbscan_output_path
    )

    print(f"Best EPS: {best_eps}, Best Min Samples: {best_min_samples}, Best Score: {best_score}")
    labels_dbscan = dbscan.clustering(best_eps, best_min_samples, reduction='pca', output=dbscan_output_path)
    dbscan.clustering(best_eps, best_min_samples, reduction='tsne', output=dbscan_output_path) """

    """print("OPTICSClustering")
    best_min_samples, best_score = optics.find_best_OPTICS(
        min_samples_range=np.arange(2, 10),
        metric='silhouette',
        plot=True,
        output=optics_output_path
    )

    print(f"Best Min Samples: {best_min_samples}, Best Score: {best_score}")
    labels_dbscan = optics.clustering(best_min_samples, reduction='pca', output=optics_output_path)
    optics.clustering(best_min_samples, reduction='tsne', output=optics_output_path)"""

