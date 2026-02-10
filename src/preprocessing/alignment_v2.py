import numpy as np
from typing import List, Dict, Any


class AngleBasedSceneAligner:
    
    def __init__(self, n_players: int = 11):
        self.n_players = n_players
    
    def compute_center_of_mass(self, positions: np.ndarray) -> np.ndarray:
        return np.mean(positions, axis=0)
    
    def compute_angles(self, positions: np.ndarray, center: np.ndarray) -> np.ndarray:
        """
        Compute angles of players around center of mass.
        
        Args:
            positions: Array of shape (N, 2) with x,y coordinates
            center: Center of mass as (2,) array
            
        Returns:
            Angles in radians, shape (N,)
        """
        # Vectors from center to each position
        vectors = positions - center
        
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])
        
        return angles
    
    def order_players_by_angle(self, players: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Order players by angle around their center of mass.
        
        Args:
            players: List of player dicts with 'x' and 'y' keys
            
        Returns:
            Ordered list of players (sorted by angle)
        """
        if len(players) == 0:
            return []
        
        positions = np.array([[p['x'], p['y']] for p in players])
        
        center = self.compute_center_of_mass(positions)
        
        angles = self.compute_angles(positions, center)
        
        sorted_indices = np.argsort(angles)
        
        return [players[i] for i in sorted_indices]
    
    def align_frame(self, frame: Dict[str, Any]) -> np.ndarray:
        """
        Align a single frame using angle-based ordering.
        
        Args:
            frame: Frame dict with 'attacking_players' key
            
        Returns:
            Array of shape (n_players, 2) with ordered positions
            Missing players filled with NaN
        """
        players = frame.get('attacking_players', [])
        
        aligned_positions = np.full((self.n_players, 2), np.nan)
        
        if len(players) == 0:
            return aligned_positions
        
        ordered_players = self.order_players_by_angle(players)
        
        for i, player in enumerate(ordered_players[:self.n_players]):
            aligned_positions[i] = [player['x'], player['y']]
        
        return aligned_positions
    
    def align_scene(self, scene: Dict[str, Any]) -> Dict[str, Any]:
        frames = scene.get('scene_frames', [])
        T = len(frames)
        
        attacking_tensor = np.full((self.n_players, 2, T), np.nan)
        defending_tensor = np.full((self.n_players, 2, T), np.nan)
        
        for t, frame in enumerate(frames):
            attacking_players = frame.get('attacking_players', [])
            attacking_aligned = self.align_frame_from_players(attacking_players)
            attacking_tensor[:, :, t] = attacking_aligned
            
            defending_players = frame.get('defending_players', [])
            defending_aligned = self.align_frame_from_players(defending_players)
            defending_tensor[:, :, t] = defending_aligned
        
        scene_tensor = np.concatenate([attacking_tensor, defending_tensor], axis=0)
        
        return {
            'scene_tensor': scene_tensor,
            'event_metadata': scene.get('event_metadata', {})
        }
    
    def align_frame_from_players(self, players: List[Dict[str, float]]) -> np.ndarray:
        aligned_positions = np.full((self.n_players, 2), np.nan)
        
        if len(players) == 0:
            return aligned_positions
        
        ordered_players = self.order_players_by_angle(players)
        
        for i, player in enumerate(ordered_players[:self.n_players]):
            aligned_positions[i] = [player['x'], player['y']]
        
        return aligned_positions
    
    def align_scenes(self, scenes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        aligned_scenes = []
        
        for i, scene in enumerate(scenes):
            aligned = self.align_scene(scene)
            aligned_scenes.append(aligned)
            
            if (i + 1) % 10 == 0:
                print(f"Aligned {i + 1}/{len(scenes)} scenes...")
        
        return aligned_scenes


def main():
    import pickle
    
    test_scene = {
        'scene_frames': [
            {
                'attacking_players': [
                    {'x': 10, 'y': 5, 'jerseyNum': 1},
                    {'x': -10, 'y': 5, 'jerseyNum': 2},
                    {'x': 10, 'y': -5, 'jerseyNum': 3},
                    {'x': -10, 'y': -5, 'jerseyNum': 4},
                    {'x': 0, 'y': 0, 'jerseyNum': 5},
                ]
            }
        ],
        'event_metadata': {'game_id': 'test', 'game_event_id': '123'}
    }
    
    # Test alignment
    aligner = AngleBasedSceneAligner(n_players=11)
    aligned = aligner.align_scene(test_scene)
    
    print("Aligned scene shape:", aligned['scene_tensor'].shape)
    print("Frame 0 positions:")
    print(aligned['scene_tensor'][:, :, 0])
    
    # Compute center and angles for visualization
    players = test_scene['scene_frames'][0]['attacking_players']
    positions = np.array([[p['x'], p['y']] for p in players])
    center = aligner.compute_center_of_mass(positions)
    angles = aligner.compute_angles(positions, center)
    
    print(f"\nCenter of mass: {center}")
    print("Angles (degrees):")
    for i, (p, angle) in enumerate(zip(players, angles)):
        print(f"  Player {p['jerseyNum']}: {np.degrees(angle):.1f}°")
    
    print("\nOrdered by angle:")
    ordered = aligner.order_players_by_angle(players)
    for p in ordered:
        print(f"  Player {p['jerseyNum']}")


if __name__ == '__main__':
    main()
