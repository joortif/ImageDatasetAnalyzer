# ImageDatasetAnalyzer

*ImageDatasetAnalyzer* is a Python library designed to simplify and automate the analysis of a set of images and optionally its segmentation labels. It provides several tools and methods to perform an initial analysis of the images and its labels obtaining useful information such as sizes, number of classes, total number of objects from a class per image and bounding boxes metrics. 

Aditionally, it includes a wide variety of models for image feature extraction and embedding of images from frameworks such as HuggingFace or PyTorch. These embeddings are useful for pattern recognition in images using traditional clustering algorithms like KMeans or AgglomerativeClustering. 

It can also be used to apply these clustering methods for [Active Learning](https://en.wikipedia.org/wiki/Active_learning_(machine_learning)) in semantic segmentation and perform a reduction of the original dataset obtaining the most representative images from each cluster. By these means, this library can be a useful tool to select which images to label for semantic segmentation (or other task that benefits from selective labeling).

## 🔧 Key features

* **Image and label dataset analysis**: Evaluate the distribution of images and labels in a dataset to understand its structure and characteristics. This analyisis can also be used ensure that everything is correct: each image has its label, sizes are accurate, the number of classes matches expectations...
* **Embedding clustering**: Group similar images using clustering techniques based on embeddings generated by pre-trained models. The library supports KMeans, AgglomerativeClustering, DBSCAN and OPTICS from skicit-learn. They also include methods to search for hyperparameter tuning using grid search.
* **Support for pre-trained models**: Compatible with embedding models from [🤗HuggingFace🤗](https://huggingface.co/), [PyTorch](https://pytorch.org/), [TensorFlow](https://www.tensorflow.org/) and [OpenCV](https://opencv.org/) frameworks. New frameworks can be easily added using the Embedding superclass.
* **Image dataset reduction**: Reduce the number of images in the dataset by selecting the most representative ones (those who are closest to the centroid) or the most diverse ones (those who are farthest from the centroid) from each cluster.   

## 🚀 Getting Started

To start using this package, install it using `pip`:

For example for Ubuntu use:
```bash
pip3 install ImageDatasetAnalyzer
```

On Windows, use:
```bash
pip install ImageDatasetAnalyzer
```

## 👩‍💻 Usage
This package includes 3 main modules for **Analysis**, **Embedding generation and Clustering** and **Dataset Reduction**.

### 📊 Dataset analysis
You can analyze the dataset and explore its properties, obtain metrics and visualizations. This module works both for image datasets with labels and for just image datasets.

```python
from imagedatasetanalyzer.src.datasets.imagelabeldataset import ImageLabelDataset

# Define paths to the images and labels
img_dir = r"images/path"
labels_dir = r"labels/path"

# Load the image and label dataset
dataset = ImageLabelDataset(img_dir=img_dir, label_dir=labels_dir)

# Alternatively, you can use just an image dataset without labels
image_dataset = ImageDataset(img_dir=img_dir)

# Perform dataset analysis (visualize and analyze)
dataset.analyze(plot=True, output="results/path", verbose=True)

# If you use only images (without labels), the analysis will provide less information
image_dataset.analyze()
```

### 🔍 Embedding generation and clustering
This module is used to generate embeddings for your images and then perform clustering using different algorithms (e.g., K-Means, DBSCAN). Here’s how to generate embeddings and perform clustering:

```python
from imagedatasetanalyzer.src.embeddings.huggingfaceembedding import HuggingFaceEmbedding
from imagedatasetanalyzer.src.datasets.imagedataset import ImageDataset
from imagedatasetanalyzer.src.models.kmeansclustering import KMeansClustering
import numpy as np

# Define image dataset directory
img_dir = r"image/path"

# Load the dataset
dataset = ImageDataset(img_dir)

# Choose an embedding model (e.g., HuggingFace DINO).
embedding_model = HuggingFaceEmbedding("facebook/dino-vits16")
embeddings = embedding_model.generate_embeddings(dataset)

# Perform K-Means clustering
kmeans = KMeansClustering(dataset, embeddings, random_state=123)
best_k = kmeans.find_elbow(25)  # Find the optimal number of clusters using the elbow method

# Apply K-Means clustering with the best number of clusters
labels_kmeans = kmeans.clustering(best_k)

# Display images from each cluster
for cluster in np.unique(labels_kmeans):
    kmeans.show_cluster_images(cluster, labels_kmeans)

# Visualize clusters using TSNE instead of PCA
kmeans.clustering(num_clusters=best_k, reduction='tsne', output='tsne_reduction')
```

### 📉 Dataset reduction 
This feature allows reducing a dataset based on various clustering methods. You can use different clustering techniques to select a smaller subset of images from the dataset. It can be done selecting those images that are closer to the centroid of each cluster (```selection_type=representative```), selecting those that are farthest (```selection_type=diverse```) or randomly (```selection_type=random```).

```python
from imagedatasetanalyzer.src.datasets.imagedataset import ImageDataset
from imagedatasetanalyzer.src.embeddings.tensorflowembedding import TensorflowEmbedding
from imagedatasetanalyzer.src.models.kmeansclustering import KMeansClustering

# Define paths
img_dir = r"images/path"

# Load dataset
dataset = ImageDataset(img_dir)

# Choose embedding method. We are using MobileNetV2 from Tensorflow.
emb = TensorflowEmbedding("MobileNetV2")
embeddings = emb.generate_embeddings(dataset)

# Initialize KMeans clustering
kmeans = KMeansClustering(dataset, embeddings, random_state=123)

# Select the number of clusters with KMeans that maximize the silhouette score.
best_k = kmeans.find_best_n_clusters(range(2,25), 'silhouette', plot=False)

# Reduce dataset using the best KMeans model according to the silhouette score. In this case, we are mantaining the 70% of the original dataset (reduction=0.7),
# obtaining the closest images from each cluster (selection_type='representative'), and ensuring that 20% of the selected images within each cluster are diverse (diverse_percentage=0.2).
# The reduced dataset will be saved to the specified output directory ("reduced/dataset/path")
reduced_dataset = kmeans.select_balanced_images(n_clusters=best_k, reduction=0.7, selection_type='representative', diverse_percentage=0.2, output="reduced/dataset/path")

# Analyze reduced dataset
reduced_dataset.analyze(plot=True, output="path/to/kmeans/output")
```

## 🧰 Requirements

## ✉️ Contact 

📧 jortizdemuruaferrero@gmail.com