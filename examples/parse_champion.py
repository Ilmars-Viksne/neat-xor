"""
This script loads a pre-trained champion genome from a pickle file
and prints its details.
"""
import argparse
import os
import pickle

def main():
    parser = argparse.ArgumentParser(
        description="Load a champion genome from a pickle file and inspect it."
    )
    parser.add_argument(
        "champion_file",
        type=str,
        help="Path to the champion genome pickle file.",
        nargs='?',
        default="darcy_results/darcy-champion.pkl"
    )
    args = parser.parse_args()

    if not os.path.exists(args.champion_file):
        print(f"Error: Champion file not found at '{args.champion_file}'")
        return

    # --- 1. Load the champion genome from pickle file ---
    print(f"Loading champion from: {args.champion_file}")
    with open(args.champion_file, "rb") as f:
        champion_genome = pickle.load(f)

    # --- 2. Print details about the champion ---
    print("\nChampion Genome Details:")
    print(f"  - Fitness: {getattr(champion_genome, 'fitness', 'N/A')}")
    print(f"  - Nodes: {len(getattr(champion_genome, 'nodes', []))}")
    print(f"  - Connections: {len(getattr(champion_genome, 'conns', []))}")

    print("\nNodes:")
    for node_id, node in getattr(champion_genome, 'nodes', {}).items():
        print(f"  - Node {node_id}: {node}")

    print("\nConnections:")
    for conn in getattr(champion_genome, 'conns', []):
        print(f"  - Connection: {conn}")


if __name__ == "__main__":
    main()
