import pickle
import os
from src.retrieval.inference import Vectorizer
from src.retrieval.index import VectorIndex
import numpy as np

class SceneSearchEngine:
    def __init__(self, model_path, scenes_path, vectors_path=None):
        with open(scenes_path, 'rb') as f:
            self.scenes = pickle.load(f)
            
        if vectors_path and os.path.exists(vectors_path):
            print(f"Loading cached vectors from {vectors_path}...")
            vectors = np.load(vectors_path)
            self.vectorizer = None 
        else:
            self.vectorizer = Vectorizer(model_path)
            vectors = self.vectorizer.vectorize_all(self.scenes)
            
        self.index = VectorIndex(dim=vectors.shape[1])
        self.index.add(vectors)
        
        self.stored_vectors = vectors
        
    def query_by_id(self, scene_idx, k=5):
        query_vec = self.stored_vectors[scene_idx].reshape(1, -1)
        sq_distances, indices = self.index.search(query_vec, k + 1)
        
        dists_flat = sq_distances.flatten()
        inds_flat = indices.flatten()
        
        results = []
        for sq_dist, idx in zip(dists_flat, inds_flat): 
            if idx == -1: continue
            if int(idx) == int(scene_idx): continue # Skip self-match
            
            results.append({
                'scene_index': int(idx),
                'similarity': float(np.sqrt(sq_dist)),
                'metadata': self.scenes[idx]['event_metadata']
            })
            
            if len(results) >= k:
                break
                
        return results
