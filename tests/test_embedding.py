import sys
import os
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'datasetanalyzerlib')))

from imagedatasetanalyzer.src.embeddings.embedding import Embedding
from imagedatasetanalyzer.src.models.kmeansclustering import KMeansClustering
from imagedatasetanalyzer.src.models.agglomerativeclustering import AgglomerativeClustering
from imagedatasetanalyzer.src.models.dbscanclustering import DBSCANClustering
from imagedatasetanalyzer.src.models.opticsclustering import OPTICSClustering
from imagedatasetanalyzer.src.datasets.imagedataset import ImageDataset
from imagedatasetanalyzer.src.embeddings.huggingfaceembedding import HuggingFaceEmbedding
from imagedatasetanalyzer.src.datasets.imagelabeldataset import ImageLabelDataset
from imagedatasetanalyzer.src.embeddings.opencvlbpembedding import OpenCVLBPEmbedding
from imagedatasetanalyzer.src.embeddings.torchembedding import PyTorchEmbedding
from imagedatasetanalyzer.src.embeddings.tensorflowembedding import TensorflowEmbedding



if __name__ == "__main__":

    def get_num_clusters(labels):

        valid_labels = labels[labels != -1]
        
        num_clusters = len(np.unique(valid_labels))
        
        return num_clusters
    
    img_dir = r"C:\Users\joortif\Desktop\datasets\Completos\crack_segmentation_dataset\images"
    labels_dir = r"C:\Users\joortif\Desktop\datasets\Completos\crack_segmentation_dataset\masks"
    output_path = r"C:\Users\joortif\Desktop\Resultados_ImageDatasetAnalyzer\crack\huggingface"
    analysis_path = r"C:\Users\joortif\Desktop\Resultados_ImageDatasetAnalyzer\crack\analysis"

    os.makedirs(os.path.join(output_path, "kmeans"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "agglomerative"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "dbscan"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "optics"), exist_ok=True)
    os.makedirs(analysis_path, exist_ok=True)


    kmeans_output_path = os.path.join(output_path, "kmeans")
    agglomerative_output_path = os.path.join(output_path, "agglomerative")
    dbscan_output_path = os.path.join(output_path, "dbscan")
    optics_output_path = os.path.join(output_path, "optics")

    os.makedirs(os.path.join(kmeans_output_path, "elbow"), exist_ok=True)
    os.makedirs(os.path.join(kmeans_output_path, "calinski"), exist_ok=True)
    os.makedirs(os.path.join(kmeans_output_path, "silhouette"), exist_ok=True)
    os.makedirs(os.path.join(kmeans_output_path, "davies"), exist_ok=True)

    kmeans_elbow_path = os.path.join(kmeans_output_path, "elbow")
    kmeans_silhouette_path = os.path.join(kmeans_output_path, "silhouette")
    kmeans_calinski_path = os.path.join(kmeans_output_path, "calinski")
    kmeans_davies_path = os.path.join(kmeans_output_path, "davies")

    os.makedirs(os.path.join(agglomerative_output_path, "calinski"), exist_ok=True)
    os.makedirs(os.path.join(agglomerative_output_path, "silhouette"), exist_ok=True)
    os.makedirs(os.path.join(agglomerative_output_path, "davies"), exist_ok=True)

    agg_silhouette_path = os.path.join(agglomerative_output_path, "silhouette")
    agg_calinski_path = os.path.join(agglomerative_output_path, "calinski")
    agg_davies_path = os.path.join(agglomerative_output_path, "davies")

    os.makedirs(os.path.join(dbscan_output_path, "calinski"), exist_ok=True)
    os.makedirs(os.path.join(dbscan_output_path, "silhouette"), exist_ok=True)
    os.makedirs(os.path.join(dbscan_output_path, "davies"), exist_ok=True)

    dbscan_silhouette_path = os.path.join(dbscan_output_path, "silhouette")
    dbscan_calinski_path = os.path.join(dbscan_output_path, "calinski")
    dbscan_davies_path = os.path.join(dbscan_output_path, "davies")

    os.makedirs(os.path.join(optics_output_path, "calinski"), exist_ok=True)
    os.makedirs(os.path.join(optics_output_path, "silhouette"), exist_ok=True)
    os.makedirs(os.path.join(optics_output_path, "davies"), exist_ok=True)

    optics_silhouette_path = os.path.join(optics_output_path, "silhouette")
    optics_calinski_path = os.path.join(optics_output_path, "calinski")
    optics_davies_path = os.path.join(optics_output_path, "davies")

    random_state = 123

    dataset = ImageDataset(img_dir)
    label_dataset = ImageLabelDataset(img_dir=img_dir, label_dir=labels_dir, background=0)

    #print("Full dataset analysis")
    #label_dataset.analyze(plot=True, output=analysis_path, verbose=True)

    emb = HuggingFaceEmbedding("facebook/dino-vits16")
    
    #emb = PyTorchEmbedding("vgg16")
    #emb = OpenCVLBPEmbedding(radius=16, num_points=48, resize_height=224, resize_width=224)
    #emb = TensorflowEmbedding("InceptionV3")

    embeddings = emb.generate_embeddings(dataset)
    np.save('embeddings\\embeddings_crack_hf_dino.npy', embeddings)
    #embeddings = np.load('embeddings\\embeddings_pytorch_sidewalk_vgg16.npy')
    #print(embeddings.shape)

    #emb = TensorflowEmbedding("MobileNetV2")

    #embeddings = emb.generate_embeddings(dataset)
    #np.save('embeddings_tf.npy', embeddings)
    #embeddings = np.load('embeddings_tf.npy')
    #print(embeddings.shape)

    #emb = OpenCVLBPEmbedding(48, 16)
    #embeddings = emb.generate_embeddings(dataset)
    #np.save('embeddings_lbp_48_16.npy', embeddings)
    #embeddings = np.load('embeddings_lbp_48_16.npy')

    #dataset = ImageLabelDataset(img_dir, labels_dir)
    #dataset.analyze()
    #embeddings = emb.generate_embeddings(dataset)

    #np.save('embeddings_labels.npy', embeddings)
    #embeddings = np.load('embeddings.npy')
    

    kmeans = KMeansClustering(dataset, embeddings, random_state)
    agglomerative = AgglomerativeClustering(dataset, embeddings, random_state)
    dbscan = DBSCANClustering(dataset, embeddings, random_state)
    optics = OPTICSClustering(dataset, embeddings, None)

    best_k_elbow = kmeans.find_elbow(25, output=kmeans_elbow_path)
    best_k_silhouette, best_score_kmeans_silhouette = kmeans.find_best_n_clusters(range(2,25), 'silhouette', output=kmeans_silhouette_path)
    best_k_calinski, best_score_kmeans_calinski = kmeans.find_best_n_clusters(range(2,25), 'calinski', output=kmeans_calinski_path)
    best_k_davies, best_score_kmeans_davies = kmeans.find_best_n_clusters(range(2,25), 'davies', output=kmeans_davies_path)

    print("=============================================")
    

    labels_kmeans = kmeans.clustering(best_k_elbow, reduction='pca', output=kmeans_elbow_path)
    kmeans.clustering(best_k_elbow, reduction='tsne', output=kmeans_elbow_path)

    for cluster in np.unique(labels_kmeans):
        kmeans.show_cluster_images(cluster, labels_kmeans, output=kmeans_elbow_path)
    
    labels_kmeans = kmeans.clustering(best_k_silhouette, reduction='pca', output=kmeans_silhouette_path)
    kmeans.clustering(best_k_silhouette, reduction='tsne', output=kmeans_silhouette_path)

    for cluster in np.unique(labels_kmeans):
        kmeans.show_cluster_images(cluster, labels_kmeans, output=kmeans_silhouette_path)

    labels_kmeans = kmeans.clustering(best_k_calinski, reduction='pca', output=kmeans_calinski_path)
    kmeans.clustering(best_k_calinski, reduction='tsne', output=kmeans_calinski_path)

    for cluster in np.unique(labels_kmeans):
        kmeans.show_cluster_images(cluster, labels_kmeans, output=kmeans_calinski_path)

    labels_kmeans = kmeans.clustering(best_k_davies, reduction='pca', output=kmeans_davies_path)
    kmeans.clustering(best_k_davies, reduction='tsne', output=kmeans_davies_path)

    for cluster in np.unique(labels_kmeans):
        kmeans.show_cluster_images(cluster, labels_kmeans, output=kmeans_davies_path)

    print("=============================================")
    
    best_n_samples_silhouette, best_linkage_silhouette, best_score_agg_silhouette = agglomerative.find_best_agglomerative_clustering(
        n_clusters_range=range(2, 15), 
        metric='silhouette', 
        output=agg_silhouette_path
    )

    best_n_samples_calinski, best_linkage_calinski, best_score_agg_calinski = agglomerative.find_best_agglomerative_clustering(
        n_clusters_range=range(2, 15), 
        metric='calinski', 
        output=agg_calinski_path
    )

    best_n_samples_davies, best_linkage_davies, best_score_agg_davies = agglomerative.find_best_agglomerative_clustering(
        n_clusters_range=range(2, 15), 
        metric='davies', 
        output=agg_davies_path
    )

    labels_agg = agglomerative.clustering(
        num_clusters=best_n_samples_silhouette, 
        linkage=best_linkage_silhouette, 
        reduction='pca', 
        output=agg_silhouette_path
    )

    agglomerative.clustering(
        num_clusters=best_n_samples_silhouette, 
        linkage=best_linkage_silhouette, 
        reduction='tsne', 
        output=agg_silhouette_path
    )

    labels_agg = agglomerative.clustering(
        num_clusters=best_n_samples_calinski, 
        linkage=best_linkage_calinski, 
        reduction='pca', 
        output=agg_calinski_path
    )

    agglomerative.clustering(
        num_clusters=best_n_samples_calinski, 
        linkage=best_linkage_calinski, 
        reduction='tsne', 
        output=agg_calinski_path
    )

    labels_agg = agglomerative.clustering(
        num_clusters=best_n_samples_davies, 
        linkage=best_linkage_davies, 
        reduction='pca', 
        output=agg_davies_path
    )

    agglomerative.clustering(
        num_clusters=best_n_samples_davies, 
        linkage=best_linkage_davies, 
        reduction='tsne', 
        output=agg_davies_path
    )

    print("=============================================")
    best_eps_silhouette, best_min_samples_silhouette, best_score_dbscan_silhouette = dbscan.find_best_DBSCAN(
        eps_range= np.arange(0.1, 5, 0.5),
        min_samples_range=np.arange(2, 15),
        metric='silhouette',
        output=dbscan_silhouette_path
    )

    best_eps_calinski, best_min_samples_calinski, best_score_dbscan_calinski = dbscan.find_best_DBSCAN(
        eps_range= np.arange(0.1, 5, 0.5),
        min_samples_range=np.arange(2, 15),
        metric='calinski',
        output=dbscan_calinski_path
    )

    best_eps_davies, best_min_samples_davies, best_score_dbscan_davies = dbscan.find_best_DBSCAN(
        eps_range= np.arange(0.1, 5, 0.5),
        min_samples_range=np.arange(2, 15),
        metric='davies',
        output=dbscan_davies_path
    )

    labels_dbscan_silhouette = dbscan.clustering(best_eps_silhouette, best_min_samples_silhouette, reduction='pca', output=dbscan_silhouette_path)
    dbscan.clustering(best_eps_silhouette, best_min_samples_silhouette, reduction='tsne', output=dbscan_silhouette_path)

    labels_dbscan_calinski = dbscan.clustering(best_eps_calinski, best_min_samples_calinski, reduction='pca', output=dbscan_calinski_path)
    dbscan.clustering(best_eps_calinski, best_min_samples_calinski, reduction='tsne', output=dbscan_calinski_path)

    labels_dbscan_davies = dbscan.clustering(best_eps_davies, best_min_samples_davies, reduction='pca', output=dbscan_davies_path)
    dbscan.clustering(best_eps_davies, best_min_samples_davies, reduction='tsne', output=dbscan_davies_path)

    print("=============================================")

    
    best_min_samples_optics_silhouette, best_score_optics_silhouette = optics.find_best_OPTICS(
        min_samples_range=np.arange(2, 15),
        metric='silhouette',
        plot=True,
        output=optics_silhouette_path
    )

    best_min_samples_optics_calinski, best_score_optics_calinski = optics.find_best_OPTICS(
        min_samples_range=np.arange(2, 15),
        metric='calinski',
        plot=True,
        output=optics_calinski_path
    )

    best_min_samples_optics_davies, best_score_optics_davies = optics.find_best_OPTICS(
        min_samples_range=np.arange(2, 15),
        metric='davies',
        plot=True,
        output=optics_davies_path
    )


    labels_optics_silhouette = optics.clustering(best_min_samples_optics_silhouette, reduction='pca', output=optics_silhouette_path)
    optics.clustering(best_min_samples_optics_silhouette, reduction='tsne', output=optics_silhouette_path)

    labels_optics_calinski = optics.clustering(best_min_samples_optics_calinski, reduction='pca', output=optics_calinski_path)
    optics.clustering(best_min_samples_optics_calinski, reduction='tsne', output=optics_calinski_path)

    labels_optics_davies = optics.clustering(best_min_samples_optics_davies, reduction='pca', output=optics_davies_path)
    optics.clustering(best_min_samples_optics_davies, reduction='tsne', output=optics_davies_path)

    print("Printing results: ")

    print("KMeansClustering")
    print(f'Best K (elbow): {best_k_elbow}')
    print(f'Best K (silhouette score): {best_k_silhouette}, Score: {best_score_kmeans_silhouette}')
    print(f'Best K (calinski score): {best_k_calinski}, Score: {best_score_kmeans_calinski}')
    print(f'Best K (davies score): {best_k_davies}, Score: {best_score_kmeans_davies}')

    print("AgglomerativeClustering")
    print(f"Best K: {best_n_samples_silhouette}, Best Linkage: {best_linkage_silhouette}, Best Score (silhouette): {best_score_agg_silhouette}")
    print(f"Best K: {best_n_samples_calinski}, Best Linkage: {best_linkage_calinski}, Best Score (calinski): {best_score_agg_calinski}")
    print(f"Best K: {best_n_samples_davies}, Best Linkage: {best_linkage_davies}, Best Score (davies): {best_score_agg_davies}")

    print("DBSCANClustering")
    print(f"Best EPS: {best_eps_silhouette}, Best Min Samples: {best_min_samples_silhouette}, Best Score (silhouette): {best_score_dbscan_silhouette}, Num clusters (except outliers): {get_num_clusters(labels_dbscan_silhouette)}")
    print(f"Best EPS: {best_eps_calinski}, Best Min Samples: {best_min_samples_calinski}, Best Score (calinski): {best_score_dbscan_calinski}, Num clusters (except outliers): {get_num_clusters(labels_dbscan_calinski)}")
    print(f"Best EPS: {best_eps_davies}, Best Min Samples: {best_min_samples_davies}, Best Score (davies): {best_score_dbscan_davies}, Num clusters (except outliers): {get_num_clusters(labels_dbscan_davies)}")

    print("OPTICSClustering")
    print(f"Best Min Samples: {best_min_samples_optics_silhouette}, Best Score (silhouette): {best_score_optics_silhouette}, Num clusters (except outliers): {get_num_clusters(labels_optics_silhouette)}")
    print(f"Best Min Samples: {best_min_samples_optics_calinski}, Best Score (calinski): {best_score_optics_calinski}, Num clusters (except outliers): {get_num_clusters(labels_optics_calinski)}")
    print(f"Best Min Samples: {best_min_samples_optics_davies}, Best Score (davies): {best_score_optics_davies}, Num clusters (except outliers): {get_num_clusters(labels_optics_davies)}")

