"""
Trains a NEAT network for the Darcy friction factor problem using the neat_ml library.
"""
import os
import pickle
import argparse

from neat_ml.neat import NEATConfig, run_neat
from neat_ml.visualization import plot_history
from darcy_problem import DarcyProblem

def run():
    """
    Sets up the configuration and runs the NEAT algorithm.
    """
    # 0. Set up argument parser
    parser = argparse.ArgumentParser(description="Train a NEAT network for the Darcy problem.")
    parser.add_argument("--start-from", type=str, default=None,
                        help="Path to a pre-trained champion genome file (.pkl) to start evolution from.")
    args = parser.parse_args()

    # 1. Define the NEAT configuration
    # These parameters are based on the NEATConfig dataclass in neat_ml/neat.py
    config = NEATConfig(
        pop_size=200,
        max_stagnation=20,
        elitism=2,
        survival_threshold=0.3    ,
        add_node_prob=0.05,
        add_connection_prob=0.08,
        prune_connection_prob=0.02,
        # The target fitness is high because fitness is 1 / (1 + MSE)
        target_fitness=0.9999,
        allow_recurrent=False
    )

    # 2. Create an instance of the problem
    problem = DarcyProblem()

    # 3. Define a directory to save results
    output_dir = "darcy_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 4. Load a starting champion if specified
    starting_champion = None
    if args.start_from:
        if os.path.exists(args.start_from):
            with open(args.start_from, "rb") as f:
                starting_champion = pickle.load(f)
            print(f"Starting evolution from champion: {args.start_from}")
        else:
            print(f"Warning: Starting champion file not found at {args.start_from}. Starting from scratch.")


    # 5. Run the NEAT algorithm
    print("Starting NEAT evolution for the Darcy problem...")
    champion_genome, history = run_neat(
        cfg=config,
        problem=problem,
        max_generations=300,
        starting_champion=starting_champion,
        viz_dir=output_dir,
        viz_each_gen=False,  # Set to True for detailed viz, but can be slow
        save_each_gen=True  # Set to True to save the best genome of each generation
    )

    # 6. Save the results
    # Before saving, remove the unpicklable _phenotype attribute
    if champion_genome:
        champion_genome._phenotype = None
        
    # Save the champion genome
    champion_path = os.path.join(output_dir, "darcy-champion.pkl")
    with open(champion_path, "wb") as f:
        pickle.dump(champion_genome, f)
    print(f"\nChampion genome saved to {champion_path}")

    # Save the evolution history
    history_path = os.path.join(output_dir, "darcy-history.pkl")
    with open(history_path, "wb") as f:
        pickle.dump(history, f)
    print(f"Evolution history saved to {history_path}")

    # Plot and save the history
    history_plot_path = os.path.join(output_dir, "darcy_fitness_history.png")
    plot_history(history, save_path=history_plot_path)
    print(f"Fitness history plot saved to {history_plot_path}")

    # Display final stats
    if champion_genome:
        print("\n--- Evolution Summary ---")
        print(f"  - Best fitness achieved: {champion_genome.fitness:.6f}")
        print(f"  - Number of generations: {len(history.generations)}")
        print(f"  - Champion genome stats:")
        print(f"    - Nodes: {len(champion_genome.nodes)}")
        print(f"    - Connections: {len(champion_genome.conns)}")
        print("-------------------------")


if __name__ == "__main__":
    run()
