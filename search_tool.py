import os
import sys
import argparse
import random
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.retrieval.search import SceneSearchEngine
from src.utils.visualization import save_search_results_enhanced
from src.preprocessing.extract_scenes import SceneExtractor
from src.config import WINDOW_SIZE_FRAMES
from src.utils.pitch import Pitch

def get_frame_index(tracking_frames, game_event_id, possession_event_id):
    """
    Finds the frame index for the given event IDs.
    """
    extractor = SceneExtractor() # Just for normalize_id
    
    target_gid = extractor.normalize_id(game_event_id)
    target_pid = extractor.normalize_id(possession_event_id)
    
    indices = []
    for idx, frame in enumerate(tracking_frames):
        gid = extractor.normalize_id(frame.get('game_event_id'))
        pid = extractor.normalize_id(frame.get('possession_event_id'))
        
        # Match primarily on game_event_id + possession_event_id
        if gid == target_gid and pid == target_pid and target_gid != "None":
            indices.append(idx)
            
    if not indices:
        return None
        
    return indices[len(indices)//2]

def extract_raw_trajectory(tracking_frames, start_idx, end_idx, pitch, attacking_direction='R'):
    """
    Extracts raw trajectories for all players and ball.
    Returns format suitable for plot_scene_raw.
    """
    scene_data = {
        'home': {}, # jersey -> list of (x,y)
        'away': {},
        'ball': []
    }
    
    window = tracking_frames[start_idx:end_idx]
    
    for frame in window:
        home_players = frame.get('homePlayersSmoothed', frame.get('homePlayers', []))
        for p in home_players:
            pid = p.get('jerseyNum')
            if pid is None: continue
            
            x, y = pitch.normalize_coordinates(p['x'], p['y'], attacking_direction)
            
            if pid not in scene_data['home']:
                scene_data['home'][pid] = []
            scene_data['home'][pid].append((x, y))
            
        away_players = frame.get('awayPlayersSmoothed', frame.get('awayPlayers', []))
        for p in away_players:
            pid = p.get('jerseyNum')
            if pid is None: continue
            
            x, y = pitch.normalize_coordinates(p['x'], p['y'], attacking_direction)
            
            if pid not in scene_data['away']:
                scene_data['away'][pid] = []
            scene_data['away'][pid].append((x, y))
            
        ball_smoothed = frame.get('ballsSmoothed')
        if ball_smoothed and ball_smoothed.get('x') is not None and ball_smoothed.get('y') is not None:
            bx, by = pitch.normalize_coordinates(ball_smoothed['x'], ball_smoothed['y'], attacking_direction)
            scene_data['ball'].append((bx, by))
        else:
            balls = frame.get('balls', [])
            if balls and balls[0].get('x') is not None and balls[0].get('y') is not None:
                bx, by = pitch.normalize_coordinates(balls[0]['x'], balls[0]['y'], attacking_direction)
                scene_data['ball'].append((bx, by))
            else:
                if scene_data['ball']:
                    scene_data['ball'].append(scene_data['ball'][-1])
                
    return scene_data

def fetch_full_raw_scene(metadata):
    """
    Loads tracking data and extracts the raw full scene.
    """
    game_id = metadata['game_id']
    game_event_id = metadata['game_event_id']
    possession_event_id = metadata['possession_event_id']
    attacking_dir = metadata.get('attacking_direction', 'R')
    
    print(f"Fetching raw data for Game {game_id}, Event {game_event_id}...")
    
    extractor = SceneExtractor()
    tracking_frames = extractor.load_tracking_data_in_memory(game_id)
    
    if not tracking_frames:
        print("Could not load tracking frames.")
        return None
        
    shot_idx = get_frame_index(tracking_frames, game_event_id, possession_event_id)
    
    if shot_idx is None:
        print("Could not find event frame.")
        return None
        
    end_idx = shot_idx + 1
    start_idx = shot_idx - WINDOW_SIZE_FRAMES + 1
    
    if start_idx < 0: start_idx = 0
    
    pitch = Pitch()
    raw_scene = extract_raw_trajectory(tracking_frames, start_idx, end_idx, pitch, attacking_dir)
    return raw_scene

def main():
    parser = argparse.ArgumentParser(description="Search and Visualize Similar Plays")
    parser.add_argument('--query_idx', type=int, default=None, help='Index of the scene to query (default: random)')
    parser.add_argument('--top_k', type=int, default=1, help='Number of similar plays to retrieve')
    parser.add_argument('--model', type=str, default='siamese_tcn_attack.pth', help='Path to model weights')
    parser.add_argument('--scenes', type=str, default='aligned_scenes.pkl', help='Path to Aligned Scenes PKL')
    parser.add_argument('--no_vis', action='store_true', help='Disable visualization')
    parser.add_argument('--output', type=str, default='search_results.png', help='Output image filename')
    parser.add_argument('--vectors', type=str, default='vectors.npy', help='Path to cached vectors')
    
    args = parser.parse_args()

    if not os.path.exists(args.model) or not os.path.exists(args.scenes):
        print("Error: Model or Scenes file not found.")
        return

    print("Initializing Search Engine...")
    engine = SceneSearchEngine(args.model, args.scenes, vectors_path=args.vectors)
    
    if args.query_idx is None:
        args.query_idx = random.randint(0, len(engine.scenes) - 1)
        print(f"No query_idx provided. Selecting random index: {args.query_idx}")

    print(f"\nQuerying for Scene {args.query_idx}...")
    
    if args.query_idx < 0 or args.query_idx >= len(engine.scenes):
        print(f"Error: Query index out of bounds (0-{len(engine.scenes)-1})")
        return

    query_meta = engine.scenes[args.query_idx]['event_metadata']
    print(f"Query Scene: Game {query_meta['game_id']}, Event {query_meta['game_event_id']}, Type {query_meta.get('type')}")
    
    results = engine.query_by_id(args.query_idx, k=args.top_k)
    
    if not results:
        print("No results found.")
        return
        
    print(f"\nTop {len(results)} Matches:")
    for i, res in enumerate(results):
        meta = res['metadata']
        print(f"{i+1}. Scene {res['scene_index']}: Dist={res['similarity']:.4f}, Game={meta['game_id']}, Event={meta['game_event_id']}")
    
    if not args.no_vis:
        print(f"\nGenerating visualization for Query and top matches...")
        
        query_raw = fetch_full_raw_scene(query_meta)
        if not query_raw:
            print("Failed to fetch query raw data. Skipping visualization.")
            return

        query_obj = {
            'raw_scene': query_raw,
            'event_metadata': query_meta
        }
        
        match_objs = []
        for res in results:
            raw = fetch_full_raw_scene(res['metadata'])
            if raw:
                match_objs.append({
                    'raw_scene': raw,
                    'distance': res['similarity'],
                    'metadata': res['metadata']
                })
            else:
                print(f"Warning: Failed to fetch raw data for match {res['scene_index']}")
        
        if match_objs:
            save_search_results_enhanced(query_obj, match_objs, output_file=args.output)
        else:
            print("No matches could be visualized.")

if __name__ == "__main__":
    main()
