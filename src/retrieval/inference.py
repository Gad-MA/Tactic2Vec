import torch
import numpy as np
from src.model import SiameseTCN

class Vectorizer:
    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        self.model = SiameseTCN(input_channels=44, hidden_channels=32, embedding_dim=64).to(self.device)
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
    def preprocess(self, scene_tensor):
        tensor = np.nan_to_num(scene_tensor, nan=0.0)
        feat = tensor.reshape(-1, tensor.shape[2])  # (44, T)
        return torch.tensor(feat, dtype=torch.float32).unsqueeze(0)

    def vectorize_all(self, scenes):
        embeddings = []
        print(f"Vectorizing {len(scenes)} scenes...")
        with torch.no_grad():
            for i, scene in enumerate(scenes):
                tensor = scene['scene_tensor']
                inp = self.preprocess(tensor).to(self.device)
                emb = self.model.forward_one(inp)
                embeddings.append(emb.cpu().numpy().flatten())
                
                if (i+1) % 100 == 0:
                    print(f"Processed {i+1}...")
                    
        return np.vstack(embeddings)
