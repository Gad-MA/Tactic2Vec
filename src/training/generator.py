import numpy as np
import random

class TrainingDataGenerator:
    def __init__(self, scenes):
        self.scenes = scenes
        self.scenes_by_type = self._group_by_type()
        
    def _group_by_type(self):
        groups = {}
        for idx, scene in enumerate(self.scenes):
            e_type = scene['event_metadata'].get('type')
            if e_type not in groups:
                groups[e_type] = []
            groups[e_type].append(idx)
        return groups

    def calculate_distance(self, idx_a, idx_b):
        tensor_a = self.scenes[idx_a]['scene_tensor'] # (11, 2, T)
        tensor_b = self.scenes[idx_b]['scene_tensor'] # (11, 2, T)
        
        # Check shapes
        if tensor_a.shape != tensor_b.shape:
             # Handle length mismatch (rare if fixed window)
             # If T differs, truncate to min
             T = min(tensor_a.shape[2], tensor_b.shape[2])
             tensor_a = tensor_a[:, :, :T]
             tensor_b = tensor_b[:, :, :T]

        tensor_a_clean = np.nan_to_num(tensor_a, nan=0.0)
        tensor_b_clean = np.nan_to_num(tensor_b, nan=0.0)
        
        distance = np.linalg.norm(tensor_a_clean - tensor_b_clean)
        
        return distance

    def generate_pairs(self, n_pairs):
        pairs = []
        types = list(self.scenes_by_type.keys())
        
        print(f"Generating {n_pairs} pairs from types: {types}...")
        
        while len(pairs) < n_pairs:
            # Pick a type
            t = random.choice(types)
            candidates = self.scenes_by_type[t]
            
            if len(candidates) < 2:
                continue
                
            i, j = random.sample(candidates, 2)
            
            dist = self.calculate_distance(i, j)
            
            pairs.append({
                'index_a': i, 
                'index_b': j,
                'distance': dist,
                'type': t
            })
            
            if len(pairs) % 100 == 0:
                print(f"Generated {len(pairs)} pairs...")
                
        return pairs
