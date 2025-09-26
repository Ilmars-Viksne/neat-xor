"""
This script provides an example of how to use the `neat_ml` library
to train a neural network for the XOR problem without using the CLI.
"""
from datetime import datetime, timezone
import os

from neat_ml import (
    NEATConfig,
    run_neat,
    save_genome_json,
    plot_history,
    decode_to_network,
    visualize_genome
)
from xor_problem import XORProblem

def main():
    # --- 1. Define the problem ---
    problem = XORProblem(use_bias_as_input=True)

    # --- 2. Configure the NEAT algorithm ---
    config = NEATConfig(
        pop_size=150,
        # The library will automatically set n_inputs and n_outputs from the problem.
        # You can customize any other parameter here.
        # For example:
        # compatibility_threshold=3.5,
        # add_node_prob=0.04,
        node_activation_options=['relu', 'tanh'],
        random_seed=42,
    )

    # --- 3. Run the evolution ---
    print("Starting NEAT evolution for the XOR problem...")

    # Define output directories
    artifacts_dir = "artifacts"
    viz_dir = "viz"
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    champion, history = run_neat(
        cfg=config,
        problem=problem,
        max_generations=100,  # A lower number for a quick example run
        viz_dir=viz_dir,
        viz_each_gen=True,
    )

    if champion is None:
        print("\nEvolution did not produce a champion.")
        return

    print(f"\nEvolution complete! Champion fitness: {champion.fitness:.4f}")

    # --- 4. Save the results ---
    # Save the champion genome
    champion_path = os.path.join(artifacts_dir, "xor_champion.json")
    save_genome_json(
        champion,
        champion_path,
        meta={
            "description": "Champion genome for the XOR problem.",
            "problem": "XORProblem",
            "source_script": os.path.basename(__file__),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
    )

    # Save the fitness history plot
    history_path = os.path.join(artifacts_dir, "xor_fitness_history.png")
    plot_history(history, save_path=history_path)

    # Save a visualization of the champion's topology
    champion_viz_path = os.path.join(viz_dir, "final_xor_champion.png")
    visualize_genome(champion, champion_viz_path, title="XOR Champion")

    # --- 5. Demonstrate the champion's performance ---
    net = decode_to_network(champion)
    print("\nChampion's performance on XOR patterns:")
    for x, y_true in problem.demo_samples():
        y_pred = net(x)[0]
        print(f"  Input: {x}, Target: {y_true[0]}, Predicted: {y_pred:.4f}")

if __name__ == "__main__":
    main()