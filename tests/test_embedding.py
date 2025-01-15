import sys
import os
import torch
import numpy as np


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'datasetanalyzerlib')))

from datasetanalyzerlib.image_similarity.embeddings.embedding import Embedding
from datasetanalyzerlib.image_similarity.models.kmeansclustering import KMeansClustering
from datasetanalyzerlib.image_similarity.models.agglomerativeclustering import AgglomerativeClustering
from datasetanalyzerlib.image_similarity.models.dbscanclustering import DBSCANClustering
from datasetanalyzerlib.image_similarity.models.opticsclustering import OPTICSClustering
from datasetanalyzerlib.image_similarity.datasets.imagedataset import ImageDataset
from datasetanalyzerlib.image_similarity.datasets.imagelabeldataset import ImageLabelDataset



if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb = Embedding("microsoft/swin-tiny-patch4-window7-224")
    img_dir = r"C:\Users\joortif\Desktop\datasets\forest_orig_reduc\train\images"
    labels_dir = r"C:\Users\joortif\Desktop\datasets\forest_orig_reduc\train\labels"
    output_path = r"C:\Users\joortif\Desktop\datasets\freiburg_resultados_labels"

    kmeans_output_path = os.path.join(output_path, "kmeans")
    agglomerative_output_path = os.path.join(output_path, "agglomerative")
    dbscan_output_path = os.path.join(output_path, "dbscan")
    optics_output_path = os.path.join(output_path, "optics")

    kmeans_elbow_path = os.path.join(kmeans_output_path, "elbow")
    kmeans_silhouette_path = os.path.join(kmeans_output_path, "calinski")

    random_state = 123

    #dataset = ImageDataset(dir)
    dataset = ImageLabelDataset(img_dir, labels_dir)
    dataset.analyze()
    #embeddings = emb.generate_embeddings(dataset)

    #np.save('embeddings_labels.npy', embeddings)
    """embeddings = np.load('embeddings_labels.npy')
    

    kmeans = KMeansClustering(dataset, embeddings, random_state)
    agglomerative = AgglomerativeClustering(dataset, embeddings, random_state)
    dbscan = DBSCANClustering(dataset, embeddings, random_state)
    optics = OPTICSClustering(dataset, embeddings, None)

    best_k_elbow = kmeans.find_elbow(25, output=kmeans_elbow_path)
    best_k_silhouette, score = kmeans.find_best_n_clusters(range(2,25), 'calinski', output=kmeans_silhouette_path)
    print("=============================================")
    print("KMeansClustering")
    print(f'Best K (elbow): {best_k_elbow}')
    print(f'Best K (calinski score): {best_k_silhouette}, Score: {score}')
    labels_kmeans = kmeans.clustering(best_k_elbow, reduction='pca', output=kmeans_elbow_path)
    kmeans.clustering(best_k_elbow, reduction='tsne', output=kmeans_elbow_path)

    for cluster in np.unique(labels_kmeans):
        kmeans.show_cluster_images(cluster, labels_kmeans, output=kmeans_elbow_path)
    
    labels_kmeans = kmeans.clustering(best_k_silhouette, reduction='pca', output=kmeans_silhouette_path)
    kmeans.clustering(best_k_silhouette, reduction='tsne', output=kmeans_silhouette_path)

    for cluster in np.unique(labels_kmeans):
        kmeans.show_cluster_images(cluster, labels_kmeans, output=kmeans_silhouette_path)
    
    #reduced_dataset_kmeans = kmeans.select_balanced_images(4, 0.7, diverse_percentage=0.5)
    #reduced_dataset_agg = agglomerative.select_balanced_images(2, 'ward', 0.7, diverse_percentage=0.5)
    
    #dbscan.clustering(6.551724137931035, 18, reduction='tsne', output=dbscan_output_path)
    #optics.clustering(12, output=optics_output_path)
    
    #reduced_dataset_dbscan = dbscan.select_balanced_images(6.551724137931035, 18, 0.7, diverse_percentage=0.5)
    #reduced_dataset_optics = optics.select_balanced_images(12, 0.7, diverse_percentage=0.5)

    print("=============================================")
    
    best_k, best_linkage, best_score = agglomerative.find_best_agglomerative_clustering(
        n_clusters_range=range(2, 15), 
        metric='calinski', 
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
    print("=============================================")
    print("DBSCANClustering")
    best_eps, best_min_samples, best_score = dbscan.find_best_DBSCAN(
        eps_range= np.linspace(5, 20, 30),
        min_samples_range=np.arange(2, 10),
        metric='calinski'
    )

    print(f"Best EPS: {best_eps}, Best Min Samples: {best_min_samples}, Best Score: {best_score}")
    labels_dbscan = dbscan.clustering(best_eps, best_min_samples, reduction='pca', output=dbscan_output_path)
    dbscan.clustering(best_eps, best_min_samples, reduction='tsne', output=dbscan_output_path)

    print("OPTICSClustering")
    best_min_samples, best_score = optics.find_best_OPTICS(
        min_samples_range=np.arange(2, 10),
        metric='calinski',
        plot=True,
        output=optics_output_path
    )

    print(f"Best Min Samples: {best_min_samples}, Best Score: {best_score}")
    labels_dbscan = optics.clustering(best_min_samples, reduction='pca', output=optics_output_path)
    optics.clustering(best_min_samples, reduction='tsne', output=optics_output_path)"""

