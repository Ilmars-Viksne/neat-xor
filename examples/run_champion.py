"""
This script demonstrates how to load a pre-trained champion genome
from a JSON file and use it for inference.
"""
import argparse
import os

from neat_ml import load_genome_json, decode_to_network

def main():
    parser = argparse.ArgumentParser(
        description="Load a champion genome and run it on sample data."
    )
    parser.add_argument(
        "champion_file",
        type=str,
        help="Path to the champion genome JSON file.",
        nargs='?',
        default="artifacts/xor_champion.json"
    )
    args = parser.parse_args()

    if not os.path.exists(args.champion_file):
        print(f"Error: Champion file not found at '{args.champion_file}'")
        print("Please run 'python examples/train_xor.py' first to generate it.")
        return

    # --- 1. Load the champion genome ---
    print(f"Loading champion from: {args.champion_file}")
    champion = load_genome_json(args.champion_file)

    # --- 2. Decode the genome into a callable neural network ---
    net = decode_to_network(champion)
    print(f"Champion loaded with fitness: {champion.fitness:.4f}")

    # --- 3. Run inference with sample data ---
    # For the XOR problem, the inputs (with bias) are:
    xor_inputs = [
        ([0.0, 0.0, 1.0], 0.0),
        ([0.0, 1.0, 1.0], 1.0),
        ([1.0, 0.0, 1.0], 1.0),
        ([1.0, 1.0, 1.0], 0.0),
    ]

    print("\nRunning inference on XOR patterns:")
    for x, target in xor_inputs:
        prediction = net(x)[0]
        print(f"  Input: {x} -> Predicted: {prediction:.4f} (Target: {target})")

if __name__ == "__main__":
    main()