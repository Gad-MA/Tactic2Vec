import argparse
import pickle
import sys
import os

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.training.generator import TrainingDataGenerator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to aligned scenes pkl')
    parser.add_argument('--output', type=str, required=True, help='Path to output training pkl')
    parser.add_argument('--n_pairs', type=int, default=1000, help='Number of pairs to generate')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        return

    print(f"Loading scenes from {args.input}...")
    with open(args.input, 'rb') as f:
        scenes = pickle.load(f)

    # Filter out scenes with too much missing data?
    # For now, use all.

    generator = TrainingDataGenerator(scenes)
    training_data = generator.generate_pairs(args.n_pairs)
    
    print(f"Saving {len(training_data)} pairs to {args.output}...")
    with open(args.output, 'wb') as f:
        pickle.dump(training_data, f)
        
    # Stats
    print("\nStats:")
    dists = [p['distance'] for p in training_data]
    print(f"Distance Min: {min(dists):.4f}")
    print(f"Distance Max: {max(dists):.4f}")
    print(f"Distance Mean: {sum(dists)/len(dists):.4f}")

if __name__ == "__main__":
    main()
