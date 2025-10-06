"""
This script loads a NEAT evolution history from a pickle file
and prints its details.
"""
import argparse
import os
import pickle
import sys

# Add the source directory to the Python path
# so we can import the EvolutionHistory class
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from neat_ml.history import EvolutionHistory

def main():
    parser = argparse.ArgumentParser(
        description="Load a NEAT evolution history from a pickle file and inspect it."
    )
    parser.add_argument(
        "history_file",
        type=str,
        help="Path to the history pickle file.",
        nargs='?',
        default="darcy_results/darcy-history.pkl"
    )
    args = parser.parse_args()

    if not os.path.exists(args.history_file):
        print(f"Error: History file not found at '{args.history_file}'")
        return

    # --- 1. Load the history from pickle file ---
    print(f"Loading history from: {args.history_file}")
    with open(args.history_file, "rb") as f:
        history: EvolutionHistory = pickle.load(f)

    # --- 2. Print details about the history ---
    try:
        num_generations = len(history.generations)
        print("\nEvolution History Details:")
        print(f"  - Number of generations: {num_generations}")

        print("\n--- Per-Generation Stats ---")
        for i in range(num_generations):
            print(f"\nGeneration {history.generations[i]}:")
            print(f"  - Best Fitness (Generation): {history.gen_best[i]:.6f}")
            print(f"  - Best Fitness (Overall): {history.best_overall[i]:.6f}")
            print(f"  - Average Fitness: {history.avg[i]:.6f}")
            print(f"  - Number of Species: {history.species[i]}")

    except (AttributeError, TypeError, IndexError) as e:
        print(f"\nError parsing the history object: {e}")
        print("The structure of the pickled object may not be as expected.")
        print("Object attributes:", dir(history))


if __name__ == "__main__":
    main()
