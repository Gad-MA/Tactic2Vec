import faiss
import numpy as np

class VectorIndex:
    def __init__(self, dim=64):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        
    def add(self, vectors):
        vectors = vectors.astype(np.float32)
        self.index.add(vectors)
        print(f"Added {len(vectors)} vectors to index.")
        
    def search(self, query_vector, k=5):
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        query_vector = query_vector.astype(np.float32)
        
        distances, indices = self.index.search(query_vector, k)
        return distances[0], indices[0]
