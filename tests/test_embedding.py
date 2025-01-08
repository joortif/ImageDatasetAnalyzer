import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'datasetanalyzerlib')))

from datasetanalyzerlib.image_similarity.embedding import Embedding

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    emb = Embedding("Salesforce/blip-image-captioning-base")

    dir = r"C:\Users\joortif\Desktop\datasets\forest_orig_reduc\train\images"

    embedding = emb.generate_embedding(dir)

    print("Embeddings generado:", embedding[0])
    print(embedding.shape)
