import os
import json
import bz2
import numpy as np
from src.config import EVENT_DATA_DIR, TRACKING_DATA_DIR, WINDOW_SIZE_FRAMES
from src.utils.pitch import Pitch

class SceneExtractor:
    def __init__(self, dataset_root=None):
        self.event_dir = EVENT_DATA_DIR
        self.tracking_dir = TRACKING_DATA_DIR
        self.pitch = Pitch()

    def normalize_id(self, val):
        if val is None:
            return "None"
        try:
            return str(int(float(val)))
        except:
            return str(val)

    def load_event_data(self, game_id):
        path = os.path.join(self.event_dir, f"{game_id}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Event file not found: {path}")

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        events = []
        target_types = {'SH'}
        
        for item in data:
            p_event = item.get('possessionEvents')
            if not p_event:
                continue
                
            e_type = p_event.get('possessionEventType')
            if e_type in target_types:
                game_event_id = item.get('gameEventId')
                possession_event_id = item.get('possessionEventId')
                
                direction = item.get('stadiumMetadata', {}).get('teamAttackingDirection', 'R')
                if str(direction).lower() == 'nan':
                     # Fallback or assume R properly normalized if PFF standard?
                     # Ideally we should know which half the team is attacking.
                     # Defaulting to R for now, but flagging it. # TODO
                     direction = 'R'

                # Identify attacking team from gameEvents
                game_events = item.get('gameEvents', {})
                is_home_attacking = game_events.get('homeTeam', True)
                attacking_team = 'home' if is_home_attacking else 'away'
                
                events.append({
                    'game_id': game_id,
                    'game_event_id': game_event_id,
                    'possession_event_id': possession_event_id,
                    'type': e_type,
                    'period': item.get('period') or item.get('gameEvents', {}).get('period'),
                    'start_time': item.get('startTime'),
                    'attacking_direction': direction,
                    'attacking_team': attacking_team,
                    'gameClock': item.get('possessionEvents', {}).get('gameClock')
                })
        
        return events

    def load_tracking_data_in_memory(self, game_id):
        path = os.path.join(self.tracking_dir, f"{game_id}.jsonl.bz2")
        if not os.path.exists(path):
            path_alt = os.path.join(self.tracking_dir, f"{game_id}.jsonl")
            if os.path.exists(path_alt):
                path = path_alt
            else:
                raise FileNotFoundError(f"Tracking file not found: {path}")

        frames = []
        
        open_func = bz2.open if path.endswith('.bz2') else open
        
        print(f"Loading tracking data for {game_id} into memory (this may take a moment)...")
        try:
            with open_func(path, 'rt', encoding='utf-8') as f:
                for line in f:
                    frames.append(json.loads(line))
        except Exception as e:
            print(f"Error reading tracking file: {e}")
            return []
            
        return frames

    def extract_window(self, tracking_frames, shot_frame_idx, attacking_direction='R', attacking_team='home'):
        end_idx = shot_frame_idx + 1
        start_idx = shot_frame_idx - WINDOW_SIZE_FRAMES + 1
        
        if start_idx < 0 or end_idx > len(tracking_frames):
            return None
        
        window_frames = tracking_frames[start_idx:end_idx]
        
        scene_data = {
            'home': [],
            'away': [],
            'ball': []
        }
        
        scene_frames = []


        for frame in window_frames:
            processed_frame = {'attacking_players': [], 'ball': None}
            
            def process_players(players_list):
                res = []
                for p in players_list:
                    raw_x, raw_y = p['x'], p['y']
                    nx, ny = self.pitch.normalize_coordinates(raw_x, raw_y, attacking_direction)
                    res.append({
                        'jerseyNum': p.get('jerseyNum'),
                        'x': nx,
                        'y': ny
                    })
                return res

            if attacking_team == 'home':
                attacking_players_raw = frame.get('homePlayersSmoothed', frame.get('homePlayers', []))
                defending_players_raw = frame.get('awayPlayersSmoothed', frame.get('awayPlayers', []))
            else:
                attacking_players_raw = frame.get('awayPlayersSmoothed', frame.get('awayPlayers', []))
                defending_players_raw = frame.get('homePlayersSmoothed', frame.get('homePlayers', []))
            
            processed_frame['attacking_players'] = process_players(attacking_players_raw)
            processed_frame['defending_players'] = process_players(defending_players_raw)
            
            ball_smoothed = frame.get('ballsSmoothed')
            if ball_smoothed and ball_smoothed.get('x') is not None and ball_smoothed.get('y') is not None:
                bx, by = ball_smoothed['x'], ball_smoothed['y']
                nbx, nby = self.pitch.normalize_coordinates(bx, by, attacking_direction)
                processed_frame['ball'] = {'x': nbx, 'y': nby}
            else:
                balls = frame.get('balls', [])
                if balls and balls[0].get('x') is not None and balls[0].get('y') is not None:
                    bx, by = balls[0]['x'], balls[0]['y']
                    nbx, nby = self.pitch.normalize_coordinates(bx, by, attacking_direction)
                    processed_frame['ball'] = {'x': nbx, 'y': nby}
            
            scene_frames.append(processed_frame)
            
            
        return scene_frames

    def process_game(self, game_id):
        print(f"Processing Game {game_id}...")
        events = self.load_event_data(game_id)
        if not events:
            print("No matching events found.")
            return []
            
        
        tracking_frames = self.load_tracking_data_in_memory(game_id)
        if not tracking_frames:
            return []
        
        id_to_frame = {}
        
        print("Building frame index...")
        for idx, frame in enumerate(tracking_frames):
            gid = self.normalize_id(frame.get('game_event_id'))
            pid = self.normalize_id(frame.get('possession_event_id'))
            if gid != "None":
                key = (gid, pid)
                if key not in id_to_frame:
                    id_to_frame[key] = []
                id_to_frame[key].append(idx)
                
        extracted_scenes = []
        
        for event in events:
            gid = self.normalize_id(event.get('game_event_id'))
            pid = self.normalize_id(event.get('possession_event_id'))
            
            indices = id_to_frame.get((gid, pid))
            if not indices:
                 continue
            shot_idx = indices[len(indices)//2]
            
            attacking_dir = event.get('attacking_direction', 'R')
            attacking_team = event.get('attacking_team', 'home')
            
            scene_frames = self.extract_window(tracking_frames, shot_idx, attacking_dir, attacking_team)
            
            if scene_frames:
                extracted_scenes.append({
                    'event_metadata': event,
                    'scene_frames': scene_frames
                })
                
        print(f"Extracted {len(extracted_scenes)} scenes.")
        return extracted_scenes

if __name__ == "__main__":
    import argparse
    import pickle
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--game_id', type=str, required=True)
    parser.add_argument('--output', type=str, default='output_scenes.pkl')
    args = parser.parse_args()
    
    extractor = SceneExtractor()
    scenes = extractor.process_game(args.game_id)
    
    with open(args.output, 'wb') as f:
        pickle.dump(scenes, f)
    
    print(f"Saved to {args.output}")
