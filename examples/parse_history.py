"""
This script loads a NEAT evolution history from a pickle file
and prints its details. It can also be used to inspect the
champion genome of a specific generation.
"""
import argparse
import os
import pickle
import sys

# Add the source directory to the Python path
# so we can import the EvolutionHistory and Genome classes
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from neat_ml.history import EvolutionHistory
from neat_ml.genome import Genome # Needed for unpickling genomes


def parse_full_history(history_file_path: str):
    """Parses and prints the summary statistics for the entire evolution history."""
    print(f"Loading history from: {history_file_path}")
    with open(history_file_path, "rb") as f:
        history: EvolutionHistory = pickle.load(f)

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


def parse_generation_champion(history_file_path: str, generation_num: int):
    """Loads and inspects the champion genome of a specific generation."""
    results_dir = os.path.dirname(history_file_path)
    champion_filename = f"gen_{generation_num:03d}_champion.pkl"
    champion_path = os.path.join(results_dir, champion_filename)

    if not os.path.exists(champion_path):
        print(f"Error: Champion file for generation {generation_num} not found at '{champion_path}'")
        print("Ensure you have run the training with 'save_each_gen=True'.")
        return

    print(f"Loading champion from: {champion_path}")
    with open(champion_path, "rb") as f:
        champion_genome: Genome = pickle.load(f)

    print(f"\nChampion Genome Details for Generation {generation_num}:")
    print(f"  - Fitness: {getattr(champion_genome, 'fitness', 'N/A')}")
    print(f"  - Nodes: {len(getattr(champion_genome, 'nodes', []))}")
    print(f"  - Connections: {len(getattr(champion_genome, 'conns', []))}")

    print("\nNodes:")
    # Sort by node ID for consistent output
    for node_id, node in sorted(getattr(champion_genome, 'nodes', {}).items()):
        print(f"  - Node {node_id}: {node}")

    print("\nConnections:")
    # Sort by innovation number for consistent output
    for conn_id, conn in sorted(getattr(champion_genome, 'conns', {}).items()):
        print(f"  - Connection {conn_id}: {conn}")


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
    parser.add_argument(
        "-g", "--generation",
        type=int,
        help="Generation number to inspect. If provided, shows details for that generation's champion genome."
    )
    args = parser.parse_args()

    if not os.path.exists(args.history_file):
        print(f"Error: History file not found at '{args.history_file}'")
        return

    if args.generation is not None:
        parse_generation_champion(args.history_file, args.generation)
    else:
        parse_full_history(args.history_file)


if __name__ == "__main__":
    main()
