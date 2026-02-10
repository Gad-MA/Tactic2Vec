
import argparse
import os
import pickle
import sys
import glob

# Ensure src is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import TRACKING_DATA_DIR, DATASET_ROOT
from src.preprocessing.extract_scenes import SceneExtractor
from src.preprocessing.alignment_v2 import AngleBasedSceneAligner
from src.training.generator import TrainingDataGenerator

def run_prep(args):
    print("=== Running Pipeline: PREP Mode ===")
    
    tracking_files = glob.glob(os.path.join(TRACKING_DATA_DIR, "*.jsonl*"))
    game_ids = set()
    for f in tracking_files:
        basename = os.path.basename(f)
        # Remove extension(s)
        if basename.endswith('.jsonl.bz2'):
            game_id = basename[:-10]
        elif basename.endswith('.jsonl'):
            game_id = basename[:-6]
        else:
            continue
        game_ids.add(game_id)
        
    game_ids = list(game_ids)
    print(f"Found {len(game_ids)} games in {TRACKING_DATA_DIR}")
    
    if args.limit_games:
        game_ids = game_ids[:args.limit_games]
        print(f"Limiting to {len(game_ids)} games.")
        
    extractor = SceneExtractor()
    aligner = AngleBasedSceneAligner(n_players=11)
    
    all_aligned_scenes = []
    
    for i, game_id in enumerate(game_ids):
        print(f"\nProcessing Game {i+1}/{len(game_ids)}: {game_id}")
        try:
            scenes = extractor.process_game(game_id)
            if not scenes:
                continue
                
            aligned = aligner.align_scenes(scenes)
            all_aligned_scenes.extend(aligned)
            
        except Exception as e:
            print(f"Error processing game {game_id}: {e}")
            import traceback
            traceback.print_exc()
            
    print(f"\nTotal aligned scenes extracted: {len(all_aligned_scenes)}")
    
    if not all_aligned_scenes:
        print("No scenes extracted. Exiting.")
        return

    # Save Aligned Scenes
    output_scenes_path = args.output_scenes
    with open(output_scenes_path, 'wb') as f:
        pickle.dump(all_aligned_scenes, f)
    print(f"Saved aligned scenes to {output_scenes_path}")
    
    print("\nGenerating training pairs...")
    generator = TrainingDataGenerator(all_aligned_scenes)
    pairs = generator.generate_pairs(args.n_pairs)
    
    output_pairs_path = args.output_pairs
    with open(output_pairs_path, 'wb') as f:
        pickle.dump(pairs, f)
    print(f"Saved {len(pairs)} pairs to {output_pairs_path}")
    
    print("\nPREP Mode Complete.")
    print(f"Upload '{output_scenes_path}' and '{output_pairs_path}' to Colab for training.")


def run_index(args):
    """
    Run Indexing: Vectorize -> Build Index
    """
    print("=== Running Pipeline: INDEX Mode ===")
    
    from src.retrieval.inference import Vectorizer
    from src.retrieval.index import VectorIndex
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return
        
    if not os.path.exists(args.input_scenes):
        print(f"Error: Scenes file not found at {args.input_scenes}")
        return
        
    print(f"Loading scenes from {args.input_scenes}...")
    with open(args.input_scenes, 'rb') as f:
        scenes = pickle.load(f)
        
    print(f"Initializing Vectorizer with model {args.model_path}...")
    vectorizer = Vectorizer(model_path=args.model_path, device='cpu')
    
    vectors = vectorizer.vectorize_all(scenes)
    print(f"Generated vectors shape: {vectors.shape}")
    
    print("Building FAISS Index...")
    index = VectorIndex(dim=vectors.shape[1])
    index.add(vectors)
    
    output_vectors_path = args.output_index
    import numpy as np
    np.save(output_vectors_path, vectors)
    print(f"Saved vectors to {output_vectors_path}")
    print("Index build successful.")


def main():
    parser = argparse.ArgumentParser(description="Similar Play Pipeline")
    subparsers = parser.add_subparsers(dest='mode', required=True)
    
    # PREP Parser
    prep_parser = subparsers.add_parser('prep', help='Prepare data for training (Extract -> Align -> Pairs)')
    prep_parser.add_argument('--limit_games', type=int, default=None, help='Limit number of games to process')
    prep_parser.add_argument('--n_pairs', type=int, default=1000, help='Number of training pairs to generate')
    prep_parser.add_argument('--output_scenes', type=str, default='aligned_scenes.pkl', help='Output path for aligned scenes')
    prep_parser.add_argument('--output_pairs', type=str, default='training_pairs.pkl', help='Output path for training pairs')
    
    # INDEX Parser
    index_parser = subparsers.add_parser('index', help='Build search index from trained model')
    index_parser.add_argument('--model_path', type=str, required=True, help='Path to trained .pth model')
    index_parser.add_argument('--input_scenes', type=str, required=True, help='Path to aligned_scenes.pkl')
    index_parser.add_argument('--output_index', type=str, default='vectors.npy', help='Output path for vectors')
    
    args = parser.parse_args()
    
    if args.mode == 'prep':
        run_prep(args)
    elif args.mode == 'index':
        run_index(args)

if __name__ == "__main__":
    main()
