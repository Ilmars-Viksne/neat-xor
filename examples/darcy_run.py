"""
Loads a trained NEAT champion for the Darcy problem and evaluates its performance
by plotting its predictions against the true Moody diagram curves.
"""
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from neat_ml.neat import decode_to_network
from darcy_problem import f_colebrook # Import the analytical function

def plot_moody_comparison(net, output_dir="darcy_results"):
    """
    Generates and saves a plot comparing the NEAT network's predictions
    to the analytical Colebrook-White friction factor curves.
    """
    filename = os.path.join(output_dir, "darcy_prediction_vs_true.png")
    plt.figure(figsize=(10, 7))

    # --- Plot Analytical Curves (Ground Truth) ---
    Re_analytical = np.logspace(2.5, 8, 400)
    eps_over_D_values = [0.0, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3]

    # Use a color cycle for better visualization
    colors = plt.cm.viridis(np.linspace(0, 1, len(eps_over_D_values)))

    for i, epsD in enumerate(eps_over_D_values):
        f_true = f_colebrook(Re_analytical, epsD)
        label_true = "Smooth (True)" if epsD == 0.0 else f"ε/D = {epsD:g} (True)"
        plt.plot(Re_analytical, f_true, '--', lw=2.0, alpha=0.8, color=colors[i], label=label_true)

    # --- Plot NEAT Network Predictions ---
    for i, epsD in enumerate(eps_over_D_values):
        f_predicted = []
        re_log_inputs = np.log10(Re_analytical)

        for re_log in re_log_inputs:
            # Network inputs: [log10(Re), eps/D, bias]
            inputs = [re_log, epsD, 1.0]
            output = net(inputs)
            f_predicted.append(output[0])

        label_neat = "NEAT" if i == 0 else "_nolegend_" # Show legend only once for NEAT
        plt.plot(Re_analytical, f_predicted, lw=1.5, color=colors[i], label=label_neat)

    # --- Formatting ---
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(10**2.5, 10**8)
    plt.ylim(0.008, 0.1)
    plt.title("NEAT Prediction vs. True Moody Diagram")
    plt.xlabel("Reynolds number Re")
    plt.ylabel("Darcy friction factor f")
    plt.grid(which='both', ls=':', color='0.7')

    # Create a clean legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='black', lw=2, ls='--', label='Analytical (True)'),
        Line2D([0], [0], color='black', lw=1.5, label='NEAT Prediction')
    ]
    for i, epsD in enumerate(eps_over_D_values):
        label = "Smooth" if epsD == 0.0 else f"ε/D = {epsD:g}"
        legend_elements.append(Line2D([0], [0], color=colors[i], lw=2, label=label))

    plt.legend(handles=legend_elements, loc='best', fontsize=8)
    plt.tight_layout()

    # Save the plot
    plt.savefig(filename, dpi=300)
    print(f"\nSaved comparison plot to {filename}")
    # plt.show() # Commented out for automated runs

def run():
    """
    Loads the champion genome and runs the evaluation.
    """
    output_dir = "darcy_results"
    champion_path = os.path.join(output_dir, "darcy-champion.pkl")

    # Check if the champion file exists
    if not os.path.exists(champion_path):
        print(f"Error: Champion genome not found at '{champion_path}'")
        print("Please run 'python examples/darcy_train.py' first.")
        return

    # Load the champion genome
    with open(champion_path, "rb") as f:
        champion_genome = pickle.load(f)
    print("Loaded champion genome:")
    print(f"  - Fitness: {champion_genome.fitness:.6f}")
    print(f"  - Nodes: {len(champion_genome.nodes)}")
    print(f"  - Connections: {len(champion_genome.conns)}")

    # Decode the genome into a callable neural network
    net = decode_to_network(champion_genome)

    # Plot the comparison
    plot_moody_comparison(net, output_dir)


if __name__ == "__main__":
    run()